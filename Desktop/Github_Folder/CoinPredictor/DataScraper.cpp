// DataScraper builds a training set from Binance Data Vision 1m kline files.
//
// It reads symbols from local CSV files, discovers the first available month of
// 1m candles for each symbol, trains on the first 6 available months, then
// writes predictions for the 7th available month.
//
// Build standalone:
//   g++ -std=c++11 -DDATASCRAPER_STANDALONE DataScraper.cpp -o data_scraper
//
// Run:
//   ./data_scraper

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "MonthSevenTester.h"

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

namespace {

const std::string kDataVisionBase = "https://data.binance.vision";
const std::string kDataVisionBucket = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision";
const std::string kPrimarySymbolCsv = "qualified_crypto_risk.csv";
const std::string kFallbackSymbolCsv = "binance_buy_sell_ratio.csv";
const std::string kTrainingCsv = "kline_growth_training.csv";
const std::string kModelCsv = "kline_growth_model.csv";
const std::string kMetricsCsv = "kline_growth_metrics.csv";
const std::string kPredictionsCsv = "kline_growth_month7_predictions.csv";
const int kTrainingMonths = 6;
const int kRequiredMonths = 7;
const int kPredictionWindowMinutes = 5;
const double kGrowthThreshold = 0.05;
const int kEpochs = 8;
const double kLearningRate = 0.04;
const double kPositiveWeightCap = 50.0;

struct Candle {
    Candle()
        : openTime(0),
          open(0.0),
          high(0.0),
          low(0.0),
          close(0.0),
          volume(0.0),
          quoteVolume(0.0),
          trades(0.0),
          takerBuyBaseVolume(0.0) {}

    long long openTime;
    double open;
    double high;
    double low;
    double close;
    double volume;
    double quoteVolume;
    double trades;
    double takerBuyBaseVolume;
};

struct Sample {
    Sample() : timeOrder(0), label(0) {}

    std::string symbol;
    std::string month;
    long long timeOrder;
    std::vector<double> features;
    int label;
};

struct Scaler {
    std::vector<double> mean;
    std::vector<double> stddev;
};

struct TrainingResult {
    std::vector<double> weights;
    Scaler scaler;
    double trainAuc;
    double testAuc;
    double testAccuracy;
    double testPrecision;
    double testRecall;
    int trainRows;
    int testRows;
    int positiveRows;
    int predictedPositiveRows;
    int truePositiveRows;
};

std::string shellQuote(const std::string &text) {
    std::string quoted = "'";
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(text[i]);
        }
    }
    quoted += "'";
    return quoted;
}

std::string runCommand(const std::string &command) {
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Unable to start command: " + command);
    }

    std::string output;
    char buffer[8192];
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        output += buffer;
    }

    const int rc = pclose(pipe);
    if (rc != 0) {
        throw std::runtime_error("Command failed: " + command);
    }

    return output;
}

std::string fetchUrl(const std::string &url) {
    return runCommand("curl -fsSL --retry 2 --connect-timeout 20 --max-time 120 " + shellQuote(url));
}

bool downloadFile(const std::string &url, const std::string &path) {
    const std::string command = "curl -fL --silent --retry 2 --connect-timeout 20 --max-time 180 -o "
        + shellQuote(path) + " " + shellQuote(url);

    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return false;
    }

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
    }

    return pclose(pipe) == 0;
}

std::vector<std::string> splitCsvLine(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (c == ',' && !inQuotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(c);
        }
    }

    fields.push_back(field);
    return fields;
}

std::string csvEscape(const std::string &value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '"') {
            escaped += "\"\"";
        } else {
            escaped.push_back(value[i]);
        }
    }
    escaped += '"';
    return escaped;
}

std::string parseTagValue(const std::string &text, const std::string &tag) {
    const std::string startTag = "<" + tag + ">";
    const std::string endTag = "</" + tag + ">";
    const size_t start = text.find(startTag);
    if (start == std::string::npos) {
        return "";
    }

    const size_t valueStart = start + startTag.size();
    const size_t end = text.find(endTag, valueStart);
    if (end == std::string::npos) {
        return "";
    }

    return text.substr(valueStart, end - valueStart);
}

std::vector<std::string> parseTagValues(const std::string &text, const std::string &tag) {
    std::vector<std::string> values;
    const std::string startTag = "<" + tag + ">";
    const std::string endTag = "</" + tag + ">";
    size_t pos = 0;

    while ((pos = text.find(startTag, pos)) != std::string::npos) {
        const size_t valueStart = pos + startTag.size();
        const size_t end = text.find(endTag, valueStart);
        if (end == std::string::npos) {
            break;
        }
        values.push_back(text.substr(valueStart, end - valueStart));
        pos = end + endTag.size();
    }

    return values;
}

bool looksLikeHeader(const std::vector<std::string> &fields) {
    return !fields.empty() && fields[0].find_first_not_of("0123456789") != std::string::npos;
}

std::vector<std::string> readSymbolsFromCsv(const std::string &path) {
    std::ifstream in(path.c_str());
    if (!in) {
        return std::vector<std::string>();
    }

    std::set<std::string> symbols;
    std::string line;
    bool firstLine = true;
    int symbolColumn = 0;

    while (std::getline(in, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') {
            line.erase(line.size() - 1);
        }
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> fields = splitCsvLine(line);
        if (firstLine) {
            firstLine = false;
            for (size_t i = 0; i < fields.size(); ++i) {
                if (fields[i] == "symbol" || fields[i] == "Symbol") {
                    symbolColumn = static_cast<int>(i);
                    break;
                }
            }
            if (fields.size() > 0 && (fields[0] == "symbol" || fields[0] == "Symbol")) {
                continue;
            }
        }

        if (symbolColumn < static_cast<int>(fields.size()) && !fields[symbolColumn].empty()) {
            symbols.insert(fields[symbolColumn]);
        }
    }

    return std::vector<std::string>(symbols.begin(), symbols.end());
}

std::vector<std::string> readRequestedSymbols(const std::vector<std::string> &symbolOverrides) {
    if (!symbolOverrides.empty()) {
        std::set<std::string> unique(symbolOverrides.begin(), symbolOverrides.end());
        std::cout << "Using " << unique.size() << " symbols from command line.\n";
        return std::vector<std::string>(unique.begin(), unique.end());
    }

    std::set<std::string> merged;

    const std::vector<std::string> primarySymbols = readSymbolsFromCsv(kPrimarySymbolCsv);
    if (!primarySymbols.empty()) {
        merged.insert(primarySymbols.begin(), primarySymbols.end());
        std::cout << "Loaded " << primarySymbols.size() << " symbols from " << kPrimarySymbolCsv << ".\n";
    }

    const std::vector<std::string> fallbackSymbols = readSymbolsFromCsv(kFallbackSymbolCsv);
    if (!fallbackSymbols.empty()) {
        merged.insert(fallbackSymbols.begin(), fallbackSymbols.end());
        std::cout << "Loaded " << fallbackSymbols.size() << " symbols from " << kFallbackSymbolCsv << ".\n";
    }

    if (!merged.empty()) {
        return std::vector<std::string>(merged.begin(), merged.end());
    }

    throw std::runtime_error("No symbols found. Expected qualified_crypto_risk.csv or binance_buy_sell_ratio.csv");
}

std::string dateFromKlineKey(const std::string &symbol, const std::string &key) {
    const std::string marker = symbol + "-1m-";
    const size_t start = key.find(marker);
    if (start == std::string::npos) {
        return "";
    }

    const size_t dateStart = start + marker.size();
    if (dateStart + 10 > key.size()) {
        return "";
    }

    const std::string date = key.substr(dateStart, 10);
    if (date.size() == 10 && date[4] == '-' && date[7] == '-') {
        return date;
    }

    return "";
}

std::vector<std::string> listKlineDates(const std::string &symbol) {
    std::set<std::string> dates;
    std::string marker;
    const std::string prefix = "data/spot/daily/klines/" + symbol + "/1m/";

    while (true) {
        std::string url = kDataVisionBucket + "?prefix=" + prefix;
        if (!marker.empty()) {
            url += "&marker=" + marker;
        }

        const std::string listing = fetchUrl(url);
        const std::vector<std::string> keys = parseTagValues(listing, "Key");
        for (size_t i = 0; i < keys.size(); ++i) {
            const std::string date = dateFromKlineKey(symbol, keys[i]);
            if (!date.empty()) {
                dates.insert(date);
            }
        }

        const std::string nextMarker = parseTagValue(listing, "NextMarker");
        if (nextMarker.empty() || nextMarker == marker) {
            break;
        }
        marker = nextMarker;
    }

    return std::vector<std::string>(dates.begin(), dates.end());
}

std::vector<std::string> monthDatesFor(const std::vector<std::string> &dates, const std::string &month) {
    std::vector<std::string> monthDates;
    for (size_t i = 0; i < dates.size(); ++i) {
        if (dates[i].substr(0, 7) == month) {
            monthDates.push_back(dates[i]);
        } else if (!monthDates.empty() && dates[i].substr(0, 7) != month) {
            break;
        }
    }

    return monthDates;
}

std::vector<std::string> firstAvailableMonths(const std::vector<std::string> &dates, int count) {
    std::vector<std::string> months;
    std::set<std::string> seen;

    for (size_t i = 0; i < dates.size(); ++i) {
        const std::string month = dates[i].substr(0, 7);
        if (seen.insert(month).second) {
            months.push_back(month);
            if (static_cast<int>(months.size()) == count) {
                break;
            }
        }
    }

    return months;
}

std::string tempZipPath(const std::string &symbol, const std::string &date) {
    return "/tmp/binance-klines-" + symbol + "-" + date + ".zip";
}

std::string binanceKlineZipUrl(const std::string &symbol, const std::string &date) {
    return kDataVisionBase + "/data/spot/daily/klines/" + symbol + "/1m/"
        + symbol + "-1m-" + date + ".zip";
}

std::string binanceMonthlyKlineZipUrl(const std::string &symbol, const std::string &month) {
    return kDataVisionBase + "/data/spot/monthly/klines/" + symbol + "/1m/"
        + symbol + "-1m-" + month + ".zip";
}

bool readCandlesFromZip(const std::string &zipPath, std::vector<Candle> &candles) {
    std::string csv;
    try {
        csv = runCommand("python3 -c "
            + shellQuote("import sys, zipfile\n"
                         "with zipfile.ZipFile(sys.argv[1]) as z:\n"
                         "    name = z.namelist()[0]\n"
                         "    sys.stdout.buffer.write(z.read(name))\n")
            + " " + shellQuote(zipPath));
    } catch (const std::exception &) {
        return false;
    }

    std::istringstream lines(csv);
    std::string line;
    bool found = false;

    while (std::getline(lines, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') {
            line.erase(line.size() - 1);
        }
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> fields = splitCsvLine(line);
        if (fields.size() < 11 || looksLikeHeader(fields)) {
            continue;
        }

        Candle candle;
        candle.openTime = std::atoll(fields[0].c_str());
        candle.open = std::atof(fields[1].c_str());
        candle.high = std::atof(fields[2].c_str());
        candle.low = std::atof(fields[3].c_str());
        candle.close = std::atof(fields[4].c_str());
        candle.volume = std::atof(fields[5].c_str());
        candle.quoteVolume = std::atof(fields[7].c_str());
        candle.trades = std::atof(fields[8].c_str());
        candle.takerBuyBaseVolume = std::atof(fields[9].c_str());

        if (candle.open > 0.0 && candle.high > 0.0 && candle.low > 0.0 && candle.close > 0.0) {
            candles.push_back(candle);
            found = true;
        }
    }

    return found;
}

std::vector<Candle> downloadCandlesForDates(const std::string &symbol, const std::vector<std::string> &monthDates);

std::vector<Candle> downloadCandlesForMonth(
    const std::string &symbol,
    const std::string &month,
    const std::vector<std::string> &dailyDates) {
    std::vector<Candle> candles;
    const std::string monthlyPath = "/tmp/binance-klines-" + symbol + "-" + month + ".zip";

    if (downloadFile(binanceMonthlyKlineZipUrl(symbol, month), monthlyPath)) {
        readCandlesFromZip(monthlyPath, candles);
        std::remove(monthlyPath.c_str());
    }

    if (candles.empty()) {
        const std::vector<std::string> monthDates = monthDatesFor(dailyDates, month);
        candles = downloadCandlesForDates(symbol, monthDates);
    }

    std::sort(candles.begin(), candles.end(), [](const Candle &left, const Candle &right) {
        return left.openTime < right.openTime;
    });

    return candles;
}

std::vector<Candle> downloadCandlesForDates(const std::string &symbol, const std::vector<std::string> &monthDates) {
    std::vector<Candle> candles;

    if (monthDates.empty()) {
        return candles;
    }

    for (size_t i = 0; i < monthDates.size(); ++i) {
        const std::string path = tempZipPath(symbol, monthDates[i]);
        if (!downloadFile(binanceKlineZipUrl(symbol, monthDates[i]), path)) {
            continue;
        }

        readCandlesFromZip(path, candles);
        std::remove(path.c_str());
    }

    std::sort(candles.begin(), candles.end(), [](const Candle &left, const Candle &right) {
        return left.openTime < right.openTime;
    });

    return candles;
}

double safeRatio(double numerator, double denominator) {
    if (denominator == 0.0) {
        return 0.0;
    }
    return numerator / denominator;
}

double clipped(double value, double lower, double upper) {
    return std::max(lower, std::min(upper, value));
}

std::vector<Sample> makeSamples(const std::string &symbol, const std::string &month, const std::vector<Candle> &candles) {
    std::vector<Sample> samples;
    if (candles.size() <= static_cast<size_t>(kPredictionWindowMinutes + 6)) {
        return samples;
    }

    for (size_t i = 5; i + kPredictionWindowMinutes < candles.size(); ++i) {
        const Candle &now = candles[i];
        double futureHigh = 0.0;
        for (int forward = 1; forward <= kPredictionWindowMinutes; ++forward) {
            futureHigh = std::max(futureHigh, candles[i + forward].high);
        }

        Sample sample;
        sample.symbol = symbol;
        sample.month = month;
        sample.timeOrder = now.openTime;
        sample.label = futureHigh >= now.close * (1.0 + kGrowthThreshold) ? 1 : 0;

        sample.features.push_back(clipped(safeRatio(now.close, candles[i - 1].close) - 1.0, -1.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.close, candles[i - 3].close) - 1.0, -1.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.close, candles[i - 5].close) - 1.0, -1.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.high - now.low, now.close), 0.0, 2.0));
        sample.features.push_back(clipped(safeRatio(now.close - now.open, now.open), -1.0, 1.0));
        sample.features.push_back(std::log(1.0 + std::max(0.0, now.volume)));
        sample.features.push_back(std::log(1.0 + std::max(0.0, now.quoteVolume)));
        sample.features.push_back(std::log(1.0 + std::max(0.0, now.trades)));
        sample.features.push_back(clipped(safeRatio(now.takerBuyBaseVolume, now.volume), 0.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.volume, candles[i - 1].volume + 1e-12) - 1.0, -10.0, 10.0));

        samples.push_back(sample);
    }

    return samples;
}

void writeTrainingCsv(const std::vector<Sample> &samples) {
    std::ofstream out(kTrainingCsv.c_str());
    if (!out) {
        throw std::runtime_error("Unable to open training CSV for writing");
    }

    out << "symbol,month,open_time,label,ret_1m,ret_3m,ret_5m,range_pct,candle_return,"
        << "log_volume,log_quote_volume,log_trades,taker_buy_ratio,volume_change\n";
    out << std::setprecision(12);

    for (size_t i = 0; i < samples.size(); ++i) {
        out << csvEscape(samples[i].symbol) << ','
            << csvEscape(samples[i].month) << ','
            << samples[i].timeOrder << ','
            << samples[i].label;
        for (size_t j = 0; j < samples[i].features.size(); ++j) {
            out << ',' << samples[i].features[j];
        }
        out << '\n';
    }
}

Scaler fitScaler(const std::vector<Sample> &samples, size_t begin, size_t end) {
    Scaler scaler;
    if (begin >= end || samples.empty()) {
        return scaler;
    }

    const size_t featureCount = samples[begin].features.size();
    scaler.mean.assign(featureCount, 0.0);
    scaler.stddev.assign(featureCount, 0.0);

    for (size_t i = begin; i < end; ++i) {
        for (size_t j = 0; j < featureCount; ++j) {
            scaler.mean[j] += samples[i].features[j];
        }
    }

    const double count = static_cast<double>(end - begin);
    for (size_t j = 0; j < featureCount; ++j) {
        scaler.mean[j] /= count;
    }

    for (size_t i = begin; i < end; ++i) {
        for (size_t j = 0; j < featureCount; ++j) {
            const double delta = samples[i].features[j] - scaler.mean[j];
            scaler.stddev[j] += delta * delta;
        }
    }

    for (size_t j = 0; j < featureCount; ++j) {
        scaler.stddev[j] = std::sqrt(scaler.stddev[j] / count);
        if (scaler.stddev[j] < 1e-9) {
            scaler.stddev[j] = 1.0;
        }
    }

    return scaler;
}

double sigmoid(double value) {
    if (value < -40.0) {
        return 0.0;
    }
    if (value > 40.0) {
        return 1.0;
    }
    return 1.0 / (1.0 + std::exp(-value));
}

double predictProbability(const Sample &sample, const Scaler &scaler, const std::vector<double> &weights) {
    double score = weights.empty() ? 0.0 : weights[0];
    for (size_t j = 0; j < sample.features.size(); ++j) {
        const double scaled = (sample.features[j] - scaler.mean[j]) / scaler.stddev[j];
        score += weights[j + 1] * scaled;
    }
    return sigmoid(score);
}

double auc(const std::vector<std::pair<double, int> > &scores) {
    if (scores.empty()) {
        return 0.0;
    }

    std::vector<std::pair<double, int> > ranked(scores);
    std::sort(ranked.begin(), ranked.end());

    double rankSumPositive = 0.0;
    int positives = 0;
    int negatives = 0;
    for (size_t i = 0; i < ranked.size(); ++i) {
        if (ranked[i].second == 1) {
            rankSumPositive += static_cast<double>(i + 1);
            ++positives;
        } else {
            ++negatives;
        }
    }

    if (positives == 0 || negatives == 0) {
        return 0.0;
    }

    return (rankSumPositive - positives * (positives + 1) / 2.0)
        / (static_cast<double>(positives) * static_cast<double>(negatives));
}

void writePredictionsCsv(
    const std::vector<Sample> &samples,
    const Scaler &scaler,
    const std::vector<double> &weights) {
    std::ofstream out(kPredictionsCsv.c_str());
    if (!out) {
        throw std::runtime_error("Unable to open 7th-month predictions CSV for writing");
    }

    out << "symbol,month,open_time,label,probability,predicted\n";
    out << std::setprecision(12);
    for (size_t i = 0; i < samples.size(); ++i) {
        const double probability = predictProbability(samples[i], scaler, weights);
        out << csvEscape(samples[i].symbol) << ','
            << csvEscape(samples[i].month) << ','
            << samples[i].timeOrder << ','
            << samples[i].label << ','
            << probability << ','
            << (probability >= 0.5 ? 1 : 0) << '\n';
    }
}

TrainingResult trainModel(std::vector<Sample> trainSamples, std::vector<Sample> testSamples) {
    if (trainSamples.empty()) {
        throw std::runtime_error("No training samples were created");
    }
    if (testSamples.empty()) {
        throw std::runtime_error("No 7th-month test samples were created");
    }

    std::sort(trainSamples.begin(), trainSamples.end(), [](const Sample &left, const Sample &right) {
        if (left.timeOrder == right.timeOrder) {
            return left.symbol < right.symbol;
        }
        return left.timeOrder < right.timeOrder;
    });
    std::sort(testSamples.begin(), testSamples.end(), [](const Sample &left, const Sample &right) {
        if (left.timeOrder == right.timeOrder) {
            return left.symbol < right.symbol;
        }
        return left.timeOrder < right.timeOrder;
    });

    const size_t featureCount = trainSamples[0].features.size();
    TrainingResult result;
    result.scaler = fitScaler(trainSamples, 0, trainSamples.size());
    result.weights.assign(featureCount + 1, 0.0);
    result.trainRows = static_cast<int>(trainSamples.size());
    result.testRows = static_cast<int>(testSamples.size());
    result.positiveRows = 0;
    result.predictedPositiveRows = 0;
    result.truePositiveRows = 0;

    int trainPositives = 0;
    int trainNegatives = 0;
    for (size_t i = 0; i < trainSamples.size(); ++i) {
        if (trainSamples[i].label == 1) {
            ++trainPositives;
            ++result.positiveRows;
        } else {
            ++trainNegatives;
        }
    }
    for (size_t i = 0; i < testSamples.size(); ++i) {
        if (testSamples[i].label == 1) {
            ++result.positiveRows;
        }
    }

    const double positiveWeight = trainPositives > 0
        ? std::min(kPositiveWeightCap, static_cast<double>(trainNegatives) / static_cast<double>(trainPositives))
        : 1.0;

    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        for (size_t i = 0; i < trainSamples.size(); ++i) {
            const double probability = predictProbability(trainSamples[i], result.scaler, result.weights);
            const double classWeight = trainSamples[i].label == 1 ? positiveWeight : 1.0;
            const double error = (probability - static_cast<double>(trainSamples[i].label)) * classWeight;

            result.weights[0] -= kLearningRate * error;
            for (size_t j = 0; j < featureCount; ++j) {
                const double scaled = (trainSamples[i].features[j] - result.scaler.mean[j]) / result.scaler.stddev[j];
                result.weights[j + 1] -= kLearningRate * error * scaled;
            }
        }
    }

    std::vector<std::pair<double, int> > trainScores;
    std::vector<std::pair<double, int> > testScores;
    int truePositive = 0;
    int trueNegative = 0;
    int falsePositive = 0;
    int falseNegative = 0;

    for (size_t i = 0; i < trainSamples.size(); ++i) {
        const double probability = predictProbability(trainSamples[i], result.scaler, result.weights);
        trainScores.push_back(std::pair<double, int>(probability, trainSamples[i].label));
    }

    for (size_t i = 0; i < testSamples.size(); ++i) {
        const double probability = predictProbability(testSamples[i], result.scaler, result.weights);
        testScores.push_back(std::pair<double, int>(probability, testSamples[i].label));
        const bool predicted = probability >= 0.5;
        if (predicted) {
            ++result.predictedPositiveRows;
        }
        if (predicted && testSamples[i].label == 1) {
            ++truePositive;
        } else if (predicted && testSamples[i].label == 0) {
            ++falsePositive;
        } else if (!predicted && testSamples[i].label == 1) {
            ++falseNegative;
        } else {
            ++trueNegative;
        }
    }

    result.truePositiveRows = truePositive;
    result.trainAuc = auc(trainScores);
    result.testAuc = auc(testScores);
    result.testAccuracy = (truePositive + trueNegative + falsePositive + falseNegative) > 0
        ? static_cast<double>(truePositive + trueNegative)
            / static_cast<double>(truePositive + trueNegative + falsePositive + falseNegative)
        : 0.0;
    result.testPrecision = (truePositive + falsePositive) > 0
        ? static_cast<double>(truePositive) / static_cast<double>(truePositive + falsePositive)
        : 0.0;
    result.testRecall = (truePositive + falseNegative) > 0
        ? static_cast<double>(truePositive) / static_cast<double>(truePositive + falseNegative)
        : 0.0;

    writeTrainingCsv(trainSamples);
    writePredictionsCsv(testSamples, result.scaler, result.weights);
    return result;
}

void writeModel(const TrainingResult &result) {
    const char *const featureNames[] = {
        "ret_1m",
        "ret_3m",
        "ret_5m",
        "range_pct",
        "candle_return",
        "log_volume",
        "log_quote_volume",
        "log_trades",
        "taker_buy_ratio",
        "volume_change"
    };

    std::ofstream model(kModelCsv.c_str());
    if (!model) {
        throw std::runtime_error("Unable to open model CSV for writing");
    }

    model << "feature,weight,mean,stddev\n";
    model << std::setprecision(12);
    model << "intercept," << result.weights[0] << ",0,1\n";
    for (size_t i = 0; i < result.scaler.mean.size(); ++i) {
        model << featureNames[i] << ','
              << result.weights[i + 1] << ','
              << result.scaler.mean[i] << ','
              << result.scaler.stddev[i] << '\n';
    }

    std::ofstream metrics(kMetricsCsv.c_str());
    if (!metrics) {
        throw std::runtime_error("Unable to open metrics CSV for writing");
    }

    metrics << "train_rows,month7_test_rows,positive_rows,predicted_positive_rows,true_positive_rows,"
        << "train_auc,month7_auc,month7_accuracy,month7_success_rate,month7_recall\n";
    metrics << result.trainRows << ','
            << result.testRows << ','
            << result.positiveRows << ','
            << result.predictedPositiveRows << ','
            << result.truePositiveRows << ','
            << std::fixed << std::setprecision(6)
            << result.trainAuc << ','
            << result.testAuc << ','
            << result.testAccuracy << ','
            << result.testPrecision << ','
            << result.testRecall << '\n';
}

} // namespace

void scrapeHistoricalCoinData(const std::vector<std::string> &symbolOverrides) {
    try {
        const std::vector<std::string> symbols = readRequestedSymbols(symbolOverrides);
        std::vector<Sample> trainingSamples;
        std::vector<Sample> monthSevenSamples;

        for (size_t i = 0; i < symbols.size(); ++i) {
            try {
                const std::vector<std::string> dates = listKlineDates(symbols[i]);
                const std::vector<std::string> months = firstAvailableMonths(dates, kRequiredMonths);
                if (static_cast<int>(months.size()) < kRequiredMonths) {
                    std::cout << "Skipping " << symbols[i] << ": fewer than 7 months of 1m klines.\n";
                    continue;
                }

                size_t symbolTrainSamples = 0;
                for (int monthIndex = 0; monthIndex < kTrainingMonths; ++monthIndex) {
                    const std::vector<Candle> trainCandles = downloadCandlesForMonth(
                        symbols[i],
                        months[monthIndex],
                        dates);
                    const std::vector<Sample> trainSamples = makeSamples(symbols[i], months[monthIndex], trainCandles);
                    trainingSamples.insert(trainingSamples.end(), trainSamples.begin(), trainSamples.end());
                    symbolTrainSamples += trainSamples.size();
                }

                const std::vector<Candle> monthSevenCandles = downloadCandlesForMonth(
                    symbols[i],
                    months[kTrainingMonths],
                    dates);
                const std::vector<Sample> testSamples = makeSamples(
                    symbols[i],
                    months[kTrainingMonths],
                    monthSevenCandles);
                monthSevenSamples.insert(monthSevenSamples.end(), testSamples.begin(), testSamples.end());

                std::cout << "Processed " << (i + 1) << "/" << symbols.size()
                          << ": " << symbols[i]
                          << " train_months=" << months[0] << ".." << months[kTrainingMonths - 1]
                          << " train_samples=" << symbolTrainSamples
                          << " test_month=" << months[kTrainingMonths]
                          << " test_samples=" << testSamples.size() << '\n';
            } catch (const std::exception &error) {
                std::cerr << "\nSkipping " << symbols[i] << ": " << error.what() << '\n';
                continue;
            }
        }

        const TrainingResult result = trainModel(trainingSamples, monthSevenSamples);
        writeModel(result);
        const MonthSevenEvaluation evaluation = evaluateMonthSevenPredictions(kPredictionsCsv);

        std::cout << "Wrote " << kTrainingCsv << ", " << kPredictionsCsv << ", "
                  << kModelCsv << ", and " << kMetricsCsv << ".\n"
                  << "Train AUC: " << std::fixed << std::setprecision(4) << result.trainAuc
                  << " | Month 7 AUC: " << result.testAuc
                  << " | Success rate: " << (evaluation.successRate * 100.0) << "%"
                  << " | Accuracy: " << (result.testAccuracy * 100.0) << "%"
                  << " | Recall: " << (result.testRecall * 100.0) << "%\n";
    } catch (const std::exception &error) {
        std::cerr << "Data scraper failed: " << error.what() << '\n';
    }
}

void scrapeHistoricalCoinData() {
    scrapeHistoricalCoinData(std::vector<std::string>());
}

#ifdef DATASCRAPER_STANDALONE
int main(int argc, char **argv) {
    std::vector<std::string> symbols;
    for (int i = 1; i < argc; ++i) {
        symbols.push_back(argv[i]);
    }

    scrapeHistoricalCoinData(symbols);
    return 0;
}
#endif
