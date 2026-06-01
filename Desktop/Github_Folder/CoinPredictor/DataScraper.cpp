// DataScraper builds a training set from Binance Data Vision 1m kline files.
//
// It reads symbols from local CSV files, discovers the first available months of
// 1m candles for each symbol, creates leakage-safe features/labels, trains a
// logistic baseline on early months, tunes the prediction threshold on a
// validation month, then writes predictions for a held-out test month.
//
// Build standalone:
//   g++ -std=c++11 -DDATASCRAPER_STANDALONE DataScraper.cpp -o data_scraper
//
// Run:
//   ./data_scraper BTCUSDT --months 8

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <climits>
#include <map>
#include <queue>
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
const std::string kLogisticMetricsCsv = "kline_growth_metrics_logistic.csv";
const std::string kPredictionsCsv = "kline_growth_month7_predictions.csv";
const std::string kLogisticPredictionsCsv = "kline_growth_predictions_logistic.csv";
const int kDefaultTrainingMonths = 6;
const int kDefaultValidationMonths = 1;
const int kDefaultTestMonths = 1;
const int kDefaultPredictionWindowMinutes = 5;
const double kDefaultGrowthThreshold = 0.05;
const double kDefaultDownsideStop = 0.02;
const double kDefaultMinNetReturn = 0.0;
const int kDefaultEpochs = 8;
const double kDefaultLearningRate = 0.04;
const double kDefaultL2Regularization = 0.001;
const double kDefaultPositiveWeightCap = 50.0;
const double kDefaultFee = 0.001;
const double kDefaultSlippage = 0.0005;
const int kDefaultMinValidationTrades = 5;
const double kDefaultInitialCapital = 10000.0;
const double kDefaultMaxPositionFraction = 0.10;
const double kDefaultMaxVolumeFraction = 0.01;
const int kDefaultMaxTradesPerPeriod = 10;
const int kDefaultTradePeriodMinutes = 60;
const int kRollingLookbackMinutes = 60;
const int kDailyLookbackMinutes = 24 * 60;

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

struct ScraperOptions {
    ScraperOptions()
        : trainingMonths(kDefaultTrainingMonths),
          validationMonths(kDefaultValidationMonths),
          testMonths(kDefaultTestMonths),
          totalMonths(kDefaultTrainingMonths + kDefaultValidationMonths + kDefaultTestMonths),
          predictionWindowMinutes(kDefaultPredictionWindowMinutes),
          holdingPeriodMinutes(kDefaultPredictionWindowMinutes),
          holdingPeriodExplicit(false),
          growthThreshold(kDefaultGrowthThreshold),
          upsideTarget(kDefaultGrowthThreshold),
          downsideStop(kDefaultDownsideStop),
          minNetReturn(kDefaultMinNetReturn),
          labelMode("target_stop"),
          tiePolicy("stop_first"),
          epochs(kDefaultEpochs),
          learningRate(kDefaultLearningRate),
          l2Regularization(kDefaultL2Regularization),
          positiveWeightCap(kDefaultPositiveWeightCap),
          minValidationTrades(kDefaultMinValidationTrades),
          fee(kDefaultFee),
          slippage(kDefaultSlippage),
          initialCapital(kDefaultInitialCapital),
          maxPositionFraction(kDefaultMaxPositionFraction),
          maxVolumeFraction(kDefaultMaxVolumeFraction),
          maxTradesPerPeriod(kDefaultMaxTradesPerPeriod),
          tradePeriodMinutes(kDefaultTradePeriodMinutes),
          thresholdObjective("profit"),
          profitSafety("explore"),
          adaptiveThresholds(true),
          splitMode("fixed"),
          trainRatio(0.70),
          validationRatio(0.15),
          testRatio(0.15),
          generateOnly(false),
          selfTest(false) {
        thresholds.push_back(0.001);
        thresholds.push_back(0.002);
        thresholds.push_back(0.005);
        thresholds.push_back(0.01);
        thresholds.push_back(0.02);
        thresholds.push_back(0.05);
        thresholds.push_back(0.10);
        thresholds.push_back(0.15);
        thresholds.push_back(0.20);
        thresholds.push_back(0.30);
        thresholds.push_back(0.40);
        thresholds.push_back(0.50);
        thresholds.push_back(0.60);
        thresholds.push_back(0.70);
        thresholds.push_back(0.80);
        thresholds.push_back(0.90);
        thresholds.push_back(0.95);
        thresholds.push_back(0.99);
    }

    int requiredMonths() const {
        return trainingMonths + validationMonths + testMonths;
    }

    int trainingMonths;
    int validationMonths;
    int testMonths;
    int totalMonths;
    int predictionWindowMinutes;
    int holdingPeriodMinutes;
    bool holdingPeriodExplicit;
    double growthThreshold;
    double upsideTarget;
    double downsideStop;
    double minNetReturn;
    std::string labelMode;
    std::string tiePolicy;
    int epochs;
    double learningRate;
    double l2Regularization;
    double positiveWeightCap;
    int minValidationTrades;
    double fee;
    double slippage;
    double initialCapital;
    double maxPositionFraction;
    double maxVolumeFraction;
    int maxTradesPerPeriod;
    int tradePeriodMinutes;
    std::string thresholdObjective;
    std::string profitSafety;
    bool adaptiveThresholds;
    std::string splitMode;
    double trainRatio;
    double validationRatio;
    double testRatio;
    bool generateOnly;
    bool selfTest;
    std::vector<double> thresholds;
};

struct MonthSplit {
    MonthSplit() : train(0), validation(0), test(0) {}
    int train;
    int validation;
    int test;
};

struct Sample {
    Sample()
        : monthIndex(0),
          timeOrder(0),
          label(0),
          forwardReturn(0.0),
          tradeReturn(0.0),
          maxFutureHighReturn(0.0),
          maxFutureLowReturn(0.0),
          quoteVolume(0.0) {}

    std::string symbol;
    std::string month;
    int monthIndex;
    long long timeOrder;
    std::vector<double> features;
    int label;
    double forwardReturn;
    double tradeReturn;
    double maxFutureHighReturn;
    double maxFutureLowReturn;
    double quoteVolume;
};

struct Scaler {
    std::vector<double> mean;
    std::vector<double> stddev;
};

struct TrainingResult {
    std::vector<double> weights;
    Scaler scaler;
    double trainAuc;
    double selectedThreshold;
    int trainRows;
    int validationRows;
    int testRows;
    int positiveRows;
};

struct EvaluationMetrics {
    EvaluationMetrics()
        : rows(0),
          actualPositiveRows(0),
          predictedTrades(0),
          truePositiveRows(0),
          falsePositiveRows(0),
          trueNegativeRows(0),
          falseNegativeRows(0),
          aucScore(0.0),
          accuracy(0.0),
          precision(0.0),
          recall(0.0),
          f1(0.0),
          winRate(0.0),
          averageForwardReturn(0.0),
          medianForwardReturn(0.0),
          averageTradeReturn(0.0),
          medianTradeReturn(0.0),
          averageMaxFavorableExcursion(0.0),
          averageMaxAdverseExcursion(0.0),
          averageProfitAfterFee(0.0),
          averageProfitAfterFeeAndSlippage(0.0),
          totalProfitAfterFee(0.0),
          totalProfitAfterFeeAndSlippage(0.0),
          profitFactor(0.0),
          maxDrawdown(0.0),
          initialCapital(0.0),
          endingCapital(0.0),
          portfolioProfit(0.0),
          portfolioReturn(0.0),
          averagePositionSize(0.0),
          medianPositionSize(0.0),
          tradesPerDay(0.0),
          tradesPerMonth(0.0),
          averageProfitPerTrade(0.0),
          worstTrade(0.0),
          maxCapitalDrawdown(0.0),
          threshold(0.0) {}

    int rows;
    int actualPositiveRows;
    int predictedTrades;
    int truePositiveRows;
    int falsePositiveRows;
    int trueNegativeRows;
    int falseNegativeRows;
    double aucScore;
    double accuracy;
    double precision;
    double recall;
    double f1;
    double winRate;
    double averageForwardReturn;
    double medianForwardReturn;
    double averageTradeReturn;
    double medianTradeReturn;
    double averageMaxFavorableExcursion;
    double averageMaxAdverseExcursion;
    double averageProfitAfterFee;
    double averageProfitAfterFeeAndSlippage;
    double totalProfitAfterFee;
    double totalProfitAfterFeeAndSlippage;
    double profitFactor;
    double maxDrawdown;
    double initialCapital;
    double endingCapital;
    double portfolioProfit;
    double portfolioReturn;
    double averagePositionSize;
    double medianPositionSize;
    double tradesPerDay;
    double tradesPerMonth;
    double averageProfitPerTrade;
    double worstTrade;
    double maxCapitalDrawdown;
    double threshold;
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

bool startsWith(const std::string &value, const std::string &prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

double parseDoubleOption(const std::string &name, const std::string &value) {
    char *end = NULL;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || (end != NULL && *end != '\0')) {
        throw std::runtime_error("Invalid value for " + name + ": " + value);
    }
    return parsed;
}

int parseIntOption(const std::string &name, const std::string &value) {
    char *end = NULL;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || (end != NULL && *end != '\0')) {
        throw std::runtime_error("Invalid value for " + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

int parseMonthsOption(const std::string &value) {
    if (value == "all" || value == "ALL" || value == "0") {
        return 0;
    }
    return parseIntOption("--months", value);
}

std::vector<double> parseThresholdGrid(const std::string &text) {
    std::vector<double> thresholds;
    std::stringstream stream(text);
    std::string value;
    while (std::getline(stream, value, ',')) {
        if (!value.empty()) {
            thresholds.push_back(parseDoubleOption("--threshold-grid", value));
        }
    }
    if (thresholds.empty()) {
        throw std::runtime_error("--threshold-grid must contain at least one threshold");
    }
    std::sort(thresholds.begin(), thresholds.end());
    return thresholds;
}

std::string optionValue(
    const std::vector<std::string> &args,
    size_t &index,
    const std::string &name,
    const std::string &current) {
    const std::string equalsPrefix = name + "=";
    if (startsWith(current, equalsPrefix)) {
        return current.substr(equalsPrefix.size());
    }
    if (index + 1 >= args.size()) {
        throw std::runtime_error("Missing value for " + name);
    }
    ++index;
    return args[index];
}

void printUsage() {
    std::cout
        << "Usage: coin_predictor train [options] [SYMBOL...]\n"
        << "Options:\n"
        << "  --months N|all             Download/generate N chronological months per symbol, or all available (default 8).\n"
        << "  --split-mode MODE          fixed or ratio. ratio splits each symbol by its available month count.\n"
        << "  --train-ratio X            Ratio split training fraction (default 0.70).\n"
        << "  --validation-ratio X       Ratio split validation fraction (default 0.15).\n"
        << "  --test-ratio X             Ratio split test fraction (default 0.15).\n"
        << "  --generate-only            Write kline_growth_training.csv and skip in-memory C++ logistic training.\n"
        << "  --self-test                Run fast offline label sanity checks and exit.\n"
        << "  --train-months N           Training months for C++ logistic split (default 6).\n"
        << "  --validation-months N      Validation months for threshold tuning (default 1).\n"
        << "  --test-months N            Out-of-sample test months (default 1).\n"
        << "  --prediction-window N      Forward prediction window in minutes (default 5).\n"
        << "  --growth-threshold X       Original future-high label threshold, e.g. 0.05.\n"
        << "  --label-mode MODE          future_high or target_stop (default target_stop).\n"
        << "  --upside-target X          Target-stop upside target percent (default 0.05).\n"
        << "  --downside-stop X          Target-stop downside stop percent (default 0.02).\n"
        << "  --tie-policy MODE          Same-candle target/stop behavior: stop_first, target_first, or skip.\n"
        << "  --min-net-return X         Minimum target return after fee and slippage (default 0).\n"
        << "  --learning-rate X          Logistic learning rate (default 0.04).\n"
        << "  --epochs N                 Logistic SGD epochs (default 8).\n"
        << "  --l2 X                     Logistic L2 regularization strength (default 0.001).\n"
        << "  --positive-weight-cap X    Max positive class weight (default 50).\n"
        << "  --initial-capital X        Starting portfolio capital used by backtest sizing (default 10000).\n"
        << "  --max-position-fraction X  Max starting-capital fraction per trade (default 0.10).\n"
        << "  --max-volume-fraction X    Max candle quote-volume fraction per trade (default 0.01).\n"
        << "  --max-trades-per-period N  Max entries across all symbols per trading period (default 10).\n"
        << "  --trade-period-minutes N   Trading-period length for the entry cap (default 60).\n"
        << "  --holding-period-minutes N Cash lock duration after entry (default prediction window).\n"
        << "  --min-validation-trades N  Minimum validation trades required for a threshold (default 5).\n"
        << "  --threshold-objective NAME profit, precision, recall, or f1 (default profit).\n"
        << "  --profit-safety MODE       strict keeps no-trade if validation profit is negative; explore picks best available (default explore).\n"
        << "  --disable-adaptive-thresholds Use only the fixed threshold grid.\n"
        << "  --threshold-grid CSV       Comma-separated thresholds, e.g. 0.05,0.10,0.50,0.90.\n"
        << "  --fee X                    Per-trade fee estimate used by evaluation (default 0.001).\n"
        << "  --slippage X               Per-trade slippage estimate used by evaluation (default 0.0005).\n";
}

bool parseArguments(
    const std::vector<std::string> &args,
    ScraperOptions &options,
    std::vector<std::string> &symbols) {
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string arg = args[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else if (arg == "--months" || startsWith(arg, "--months=")) {
            options.totalMonths = parseMonthsOption(optionValue(args, i, "--months", arg));
            if (options.totalMonths == 0) {
                options.splitMode = "ratio";
            }
        } else if (arg == "--train-months" || startsWith(arg, "--train-months=")) {
            options.trainingMonths = parseIntOption("--train-months", optionValue(args, i, "--train-months", arg));
        } else if (arg == "--validation-months" || startsWith(arg, "--validation-months=")) {
            options.validationMonths = parseIntOption("--validation-months", optionValue(args, i, "--validation-months", arg));
        } else if (arg == "--test-months" || startsWith(arg, "--test-months=")) {
            options.testMonths = parseIntOption("--test-months", optionValue(args, i, "--test-months", arg));
        } else if (arg == "--split-mode" || startsWith(arg, "--split-mode=")) {
            options.splitMode = optionValue(args, i, "--split-mode", arg);
        } else if (arg == "--train-ratio" || startsWith(arg, "--train-ratio=")) {
            options.trainRatio = parseDoubleOption("--train-ratio", optionValue(args, i, "--train-ratio", arg));
        } else if (arg == "--validation-ratio" || startsWith(arg, "--validation-ratio=")) {
            options.validationRatio = parseDoubleOption("--validation-ratio", optionValue(args, i, "--validation-ratio", arg));
        } else if (arg == "--test-ratio" || startsWith(arg, "--test-ratio=")) {
            options.testRatio = parseDoubleOption("--test-ratio", optionValue(args, i, "--test-ratio", arg));
        } else if (arg == "--generate-only") {
            options.generateOnly = true;
        } else if (arg == "--self-test") {
            options.selfTest = true;
        } else if (arg == "--prediction-window" || startsWith(arg, "--prediction-window=")) {
            options.predictionWindowMinutes = parseIntOption("--prediction-window", optionValue(args, i, "--prediction-window", arg));
            if (!options.holdingPeriodExplicit) {
                options.holdingPeriodMinutes = options.predictionWindowMinutes;
            }
        } else if (arg == "--growth-threshold" || startsWith(arg, "--growth-threshold=")) {
            options.growthThreshold = parseDoubleOption("--growth-threshold", optionValue(args, i, "--growth-threshold", arg));
            options.upsideTarget = options.growthThreshold;
        } else if (arg == "--label-mode" || startsWith(arg, "--label-mode=")) {
            options.labelMode = optionValue(args, i, "--label-mode", arg);
        } else if (arg == "--upside-target" || startsWith(arg, "--upside-target=")) {
            options.upsideTarget = parseDoubleOption("--upside-target", optionValue(args, i, "--upside-target", arg));
        } else if (arg == "--downside-stop" || startsWith(arg, "--downside-stop=")) {
            options.downsideStop = parseDoubleOption("--downside-stop", optionValue(args, i, "--downside-stop", arg));
        } else if (arg == "--tie-policy" || startsWith(arg, "--tie-policy=")) {
            options.tiePolicy = optionValue(args, i, "--tie-policy", arg);
        } else if (arg == "--min-net-return" || startsWith(arg, "--min-net-return=")) {
            options.minNetReturn = parseDoubleOption("--min-net-return", optionValue(args, i, "--min-net-return", arg));
        } else if (arg == "--learning-rate" || startsWith(arg, "--learning-rate=")) {
            options.learningRate = parseDoubleOption("--learning-rate", optionValue(args, i, "--learning-rate", arg));
        } else if (arg == "--epochs" || startsWith(arg, "--epochs=")) {
            options.epochs = parseIntOption("--epochs", optionValue(args, i, "--epochs", arg));
        } else if (arg == "--l2" || startsWith(arg, "--l2=")) {
            options.l2Regularization = parseDoubleOption("--l2", optionValue(args, i, "--l2", arg));
        } else if (arg == "--positive-weight-cap" || startsWith(arg, "--positive-weight-cap=")) {
            options.positiveWeightCap = parseDoubleOption("--positive-weight-cap", optionValue(args, i, "--positive-weight-cap", arg));
        } else if (arg == "--initial-capital" || startsWith(arg, "--initial-capital=")) {
            options.initialCapital = parseDoubleOption("--initial-capital", optionValue(args, i, "--initial-capital", arg));
        } else if (arg == "--max-position-fraction" || startsWith(arg, "--max-position-fraction=")) {
            options.maxPositionFraction = parseDoubleOption("--max-position-fraction", optionValue(args, i, "--max-position-fraction", arg));
        } else if (arg == "--max-volume-fraction" || startsWith(arg, "--max-volume-fraction=")) {
            options.maxVolumeFraction = parseDoubleOption("--max-volume-fraction", optionValue(args, i, "--max-volume-fraction", arg));
        } else if (arg == "--max-trades-per-period" || startsWith(arg, "--max-trades-per-period=")) {
            options.maxTradesPerPeriod = parseIntOption("--max-trades-per-period", optionValue(args, i, "--max-trades-per-period", arg));
        } else if (arg == "--trade-period-minutes" || startsWith(arg, "--trade-period-minutes=")) {
            options.tradePeriodMinutes = parseIntOption("--trade-period-minutes", optionValue(args, i, "--trade-period-minutes", arg));
        } else if (arg == "--holding-period-minutes" || startsWith(arg, "--holding-period-minutes=")) {
            options.holdingPeriodMinutes = parseIntOption("--holding-period-minutes", optionValue(args, i, "--holding-period-minutes", arg));
            options.holdingPeriodExplicit = true;
        } else if (arg == "--cooldown-minutes" || startsWith(arg, "--cooldown-minutes=")) {
            if (parseIntOption("--cooldown-minutes", optionValue(args, i, "--cooldown-minutes", arg)) < 0) {
                throw std::runtime_error("--cooldown-minutes cannot be negative");
            }
            std::cerr << "Warning: --cooldown-minutes is retained for compatibility and ignored; "
                      << "portfolio entry limits replace cooldown.\n";
        } else if (arg == "--max-trades-per-symbol-month" || startsWith(arg, "--max-trades-per-symbol-month=")) {
            if (parseIntOption("--max-trades-per-symbol-month", optionValue(args, i, "--max-trades-per-symbol-month", arg)) < 0) {
                throw std::runtime_error("--max-trades-per-symbol-month cannot be negative");
            }
            std::cerr << "Warning: --max-trades-per-symbol-month is retained for compatibility and ignored; "
                      << "use --max-trades-per-period.\n";
        } else if (arg == "--min-validation-trades" || startsWith(arg, "--min-validation-trades=")) {
            options.minValidationTrades = parseIntOption("--min-validation-trades", optionValue(args, i, "--min-validation-trades", arg));
        } else if (arg == "--threshold-objective" || startsWith(arg, "--threshold-objective=")) {
            options.thresholdObjective = optionValue(args, i, "--threshold-objective", arg);
        } else if (arg == "--profit-safety" || startsWith(arg, "--profit-safety=")) {
            options.profitSafety = optionValue(args, i, "--profit-safety", arg);
        } else if (arg == "--disable-adaptive-thresholds") {
            options.adaptiveThresholds = false;
        } else if (arg == "--threshold-grid" || startsWith(arg, "--threshold-grid=")) {
            options.thresholds = parseThresholdGrid(optionValue(args, i, "--threshold-grid", arg));
        } else if (arg == "--fee" || startsWith(arg, "--fee=")) {
            options.fee = parseDoubleOption("--fee", optionValue(args, i, "--fee", arg));
        } else if (arg == "--slippage" || startsWith(arg, "--slippage=")) {
            options.slippage = parseDoubleOption("--slippage", optionValue(args, i, "--slippage", arg));
        } else if (startsWith(arg, "--")) {
            throw std::runtime_error("Unknown option: " + arg);
        } else {
            symbols.push_back(arg);
        }
    }

    if (options.trainingMonths <= 0 || options.validationMonths <= 0 || options.testMonths <= 0) {
        throw std::runtime_error("Train, validation, and test month counts must all be positive");
    }
    if (options.totalMonths > 0 && options.totalMonths < options.requiredMonths()) {
        options.totalMonths = options.requiredMonths();
    }
    if (options.splitMode != "fixed" && options.splitMode != "ratio") {
        throw std::runtime_error("--split-mode must be fixed or ratio");
    }
    if (options.trainRatio <= 0.0 || options.validationRatio <= 0.0 || options.testRatio <= 0.0) {
        throw std::runtime_error("--train-ratio, --validation-ratio, and --test-ratio must be positive");
    }
    const double ratioSum = options.trainRatio + options.validationRatio + options.testRatio;
    if (std::fabs(ratioSum - 1.0) > 0.001) {
        throw std::runtime_error("--train-ratio + --validation-ratio + --test-ratio must equal 1.0");
    }
    if (options.predictionWindowMinutes <= 0) {
        throw std::runtime_error("--prediction-window must be positive");
    }
    if (options.initialCapital <= 0.0) {
        throw std::runtime_error("--initial-capital must be positive");
    }
    if (options.maxPositionFraction <= 0.0 || options.maxPositionFraction > 1.0) {
        throw std::runtime_error("--max-position-fraction must be between 0 and 1");
    }
    if (options.maxVolumeFraction <= 0.0 || options.maxVolumeFraction > 1.0) {
        throw std::runtime_error("--max-volume-fraction must be between 0 and 1");
    }
    if (options.maxTradesPerPeriod <= 0 || options.tradePeriodMinutes <= 0 || options.holdingPeriodMinutes <= 0) {
        throw std::runtime_error("Portfolio period, trade cap, and holding period must be positive");
    }
    if (options.minValidationTrades < 0) {
        throw std::runtime_error("--min-validation-trades cannot be negative");
    }
    if (options.labelMode != "future_high" && options.labelMode != "target_stop") {
        throw std::runtime_error("--label-mode must be future_high or target_stop");
    }
    if (options.tiePolicy != "stop_first" && options.tiePolicy != "target_first" && options.tiePolicy != "skip") {
        throw std::runtime_error("--tie-policy must be stop_first, target_first, or skip");
    }
    if (options.minNetReturn < 0.0) {
        throw std::runtime_error("--min-net-return cannot be negative");
    }
    if (options.upsideTarget <= 0.0 || options.downsideStop <= 0.0) {
        throw std::runtime_error("--upside-target and --downside-stop must be positive");
    }
    if (options.labelMode == "target_stop" && options.upsideTarget - options.fee - options.slippage < options.minNetReturn) {
        throw std::runtime_error("--upside-target minus fee and slippage must be at least --min-net-return");
    }
    if (options.thresholdObjective == "success_rate") {
        options.thresholdObjective = "precision";
    }
    if (options.thresholdObjective != "profit"
            && options.thresholdObjective != "precision"
            && options.thresholdObjective != "recall"
            && options.thresholdObjective != "f1") {
        throw std::runtime_error("--threshold-objective must be profit, precision, recall, or f1");
    }
    if (options.profitSafety != "strict" && options.profitSafety != "explore") {
        throw std::runtime_error("--profit-safety must be strict or explore");
    }
    if (options.epochs <= 0) {
        throw std::runtime_error("--epochs must be positive");
    }
    return true;
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

        // S3 ListBucket V1 often omits NextMarker unless a delimiter is used.
        // In that case the correct continuation marker is the last key from
        // the current page; otherwise --months all stops after about 500 days
        // because each date has both a .zip and .CHECKSUM key.
        const std::string nextMarker = parseTagValue(listing, "NextMarker");
        const std::string isTruncated = parseTagValue(listing, "IsTruncated");
        if (!nextMarker.empty() && nextMarker != marker) {
            marker = nextMarker;
        } else if ((isTruncated == "true" || isTruncated == "True" || isTruncated == "1")
                && !keys.empty()
                && keys.back() != marker) {
            marker = keys.back();
        } else {
            break;
        }
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
            if (count > 0 && static_cast<int>(months.size()) == count) {
                break;
            }
        }
    }

    return months;
}

MonthSplit splitForMonthCount(int availableMonths, const ScraperOptions &options) {
    MonthSplit split;
    if (options.splitMode == "fixed") {
        split.train = options.trainingMonths;
        split.validation = options.validationMonths;
        split.test = options.testMonths;
        return split;
    }

    if (availableMonths < 3) {
        return split;
    }

    split.validation = std::max(1, static_cast<int>(std::floor(availableMonths * options.validationRatio + 0.5)));
    split.test = std::max(1, static_cast<int>(std::floor(availableMonths * options.testRatio + 0.5)));
    if (split.validation + split.test >= availableMonths) {
        split.validation = 1;
        split.test = 1;
    }
    split.train = availableMonths - split.validation - split.test;
    if (split.train < 1) {
        split.train = 1;
        if (split.validation > 1) {
            --split.validation;
        } else if (split.test > 1) {
            --split.test;
        }
    }
    return split;
}

std::string monthRangeDescription(const std::vector<std::string> &months, int start, int count) {
    if (count <= 0 || start < 0 || start >= static_cast<int>(months.size())) {
        return "none";
    }
    const int end = std::min(static_cast<int>(months.size()) - 1, start + count - 1);
    if (start == end) {
        return months[start];
    }
    return months[start] + ".." + months[end];
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

std::vector<std::string> featureNames() {
    std::vector<std::string> names;
    names.push_back("ret_1m");
    names.push_back("ret_3m");
    names.push_back("ret_5m");
    names.push_back("range_pct");
    names.push_back("candle_return");
    names.push_back("log_volume");
    names.push_back("log_quote_volume");
    names.push_back("log_trades");
    names.push_back("taker_buy_ratio");
    names.push_back("volume_change");
    names.push_back("ret_10m");
    names.push_back("ret_15m");
    names.push_back("ret_30m");
    names.push_back("ret_60m");
    names.push_back("rolling_volatility_60m");
    names.push_back("rolling_volume_mean_60m");
    names.push_back("rolling_volume_zscore_60m");
    names.push_back("relative_volume_previous_hour");
    names.push_back("rolling_trade_count_zscore_60m");
    names.push_back("taker_buy_ratio_change");
    names.push_back("taker_buy_imbalance");
    names.push_back("distance_from_recent_high_60m");
    names.push_back("distance_from_recent_low_60m");
    names.push_back("consecutive_green_candles");
    names.push_back("consecutive_red_candles");
    names.push_back("volume_acceleration");
    names.push_back("trade_count_acceleration");
    names.push_back("candle_body_range_ratio");
    names.push_back("upper_wick_pct");
    names.push_back("lower_wick_pct");
    names.push_back("distance_from_high_24h");
    names.push_back("distance_from_low_24h");
    names.push_back("rolling_quote_volume_mean_60m");
    names.push_back("rolling_quote_volume_zscore_60m");
    names.push_back("relative_quote_volume_previous_hour");
    names.push_back("taker_buy_ratio_mean_60m");
    names.push_back("taker_buy_ratio_zscore_60m");
    return names;
}

double closeReturn(const std::vector<Candle> &candles, size_t index, int minutes) {
    if (minutes <= 0 || index < static_cast<size_t>(minutes)) {
        return 0.0;
    }
    return clipped(safeRatio(candles[index].close, candles[index - minutes].close) - 1.0, -1.0, 1.0);
}

void rollingMeanStd(
    const std::vector<Candle> &candles,
    size_t begin,
    size_t end,
    bool useTrades,
    double &mean,
    double &stddev) {
    mean = 0.0;
    stddev = 0.0;
    if (begin >= end) {
        return;
    }

    for (size_t i = begin; i < end; ++i) {
        mean += useTrades ? candles[i].trades : candles[i].volume;
    }
    const double count = static_cast<double>(end - begin);
    mean /= count;

    for (size_t i = begin; i < end; ++i) {
        const double value = useTrades ? candles[i].trades : candles[i].volume;
        const double delta = value - mean;
        stddev += delta * delta;
    }
    stddev = std::sqrt(stddev / count);
    if (stddev < 1e-12) {
        stddev = 1.0;
    }
}

double rollingReturnVolatility(const std::vector<Candle> &candles, size_t index, int lookback) {
    if (index < static_cast<size_t>(lookback) || lookback <= 1) {
        return 0.0;
    }

    const size_t begin = index - static_cast<size_t>(lookback) + 1;
    double mean = 0.0;
    std::vector<double> returns;
    returns.reserve(static_cast<size_t>(lookback));
    for (size_t i = begin; i <= index; ++i) {
        const double value = safeRatio(candles[i].close, candles[i - 1].close) - 1.0;
        returns.push_back(value);
        mean += value;
    }
    mean /= static_cast<double>(returns.size());

    double variance = 0.0;
    for (size_t i = 0; i < returns.size(); ++i) {
        const double delta = returns[i] - mean;
        variance += delta * delta;
    }
    return std::sqrt(variance / static_cast<double>(returns.size()));
}

int consecutiveCandles(const std::vector<Candle> &candles, size_t index, bool green) {
    int count = 0;
    size_t cursor = index + 1;
    while (cursor > 0) {
        --cursor;
        const bool isGreen = candles[cursor].close >= candles[cursor].open;
        if (isGreen != green) {
            break;
        }
        ++count;
    }
    return count;
}

double takerBuyRatio(const Candle &candle) {
    return clipped(safeRatio(candle.takerBuyBaseVolume, candle.volume), 0.0, 1.0);
}

struct RollingWindowStats {
    RollingWindowStats() : sum(0.0), sumSquares(0.0) {}

    void push(double value, size_t maxSize) {
        values.push_back(value);
        sum += value;
        sumSquares += value * value;
        if (values.size() > maxSize) {
            const double removed = values.front();
            values.pop_front();
            sum -= removed;
            sumSquares -= removed * removed;
        }
    }

    double mean() const {
        return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
    }

    double stddev() const {
        if (values.empty()) {
            return 1.0;
        }
        const double average = mean();
        const double variance = std::max(0.0, sumSquares / static_cast<double>(values.size()) - average * average);
        const double result = std::sqrt(variance);
        return result < 1e-12 ? 1.0 : result;
    }

    std::deque<double> values;
    double sum;
    double sumSquares;
};

struct FeatureContext {
    std::vector<double> quoteVolumeMean;
    std::vector<double> quoteVolumeStddev;
    std::vector<double> takerBuyMean;
    std::vector<double> takerBuyStddev;
    std::vector<double> high24h;
    std::vector<double> low24h;
};

FeatureContext buildFeatureContext(const std::vector<Candle> &candles) {
    FeatureContext context;
    const size_t count = candles.size();
    context.quoteVolumeMean.resize(count, 0.0);
    context.quoteVolumeStddev.resize(count, 1.0);
    context.takerBuyMean.resize(count, 0.0);
    context.takerBuyStddev.resize(count, 1.0);
    context.high24h.resize(count, 0.0);
    context.low24h.resize(count, 0.0);

    RollingWindowStats quoteVolume;
    RollingWindowStats takerBuy;
    std::deque<size_t> highIndices;
    std::deque<size_t> lowIndices;
    for (size_t i = 0; i < count; ++i) {
        context.quoteVolumeMean[i] = quoteVolume.mean();
        context.quoteVolumeStddev[i] = quoteVolume.stddev();
        context.takerBuyMean[i] = takerBuy.mean();
        context.takerBuyStddev[i] = takerBuy.stddev();

        while (!highIndices.empty() && highIndices.front() + kDailyLookbackMinutes <= i) {
            highIndices.pop_front();
        }
        while (!lowIndices.empty() && lowIndices.front() + kDailyLookbackMinutes <= i) {
            lowIndices.pop_front();
        }
        while (!highIndices.empty() && candles[highIndices.back()].high <= candles[i].high) {
            highIndices.pop_back();
        }
        while (!lowIndices.empty() && candles[lowIndices.back()].low >= candles[i].low) {
            lowIndices.pop_back();
        }
        highIndices.push_back(i);
        lowIndices.push_back(i);
        context.high24h[i] = candles[highIndices.front()].high;
        context.low24h[i] = candles[lowIndices.front()].low;

        quoteVolume.push(candles[i].quoteVolume, kRollingLookbackMinutes);
        takerBuy.push(takerBuyRatio(candles[i]), kRollingLookbackMinutes);
    }
    return context;
}

std::vector<Sample> makeSamples(
    const std::string &symbol,
    const std::string &month,
    int monthIndex,
    const std::vector<Candle> &candles,
    const ScraperOptions &options) {
    std::vector<Sample> samples;
    const size_t historyStart = static_cast<size_t>(std::max(5, kRollingLookbackMinutes));
    if (candles.size() <= historyStart + static_cast<size_t>(options.predictionWindowMinutes)) {
        return samples;
    }
    const FeatureContext context = buildFeatureContext(candles);

    for (size_t i = historyStart; i + options.predictionWindowMinutes < candles.size(); ++i) {
        const Candle &now = candles[i];
        double futureHigh = 0.0;
        double futureLow = std::numeric_limits<double>::max();
        for (int forward = 1; forward <= options.predictionWindowMinutes; ++forward) {
            futureHigh = std::max(futureHigh, candles[i + forward].high);
            futureLow = std::min(futureLow, candles[i + forward].low);
        }

        Sample sample;
        sample.symbol = symbol;
        sample.month = month;
        sample.monthIndex = monthIndex;
        sample.timeOrder = now.openTime;
        sample.forwardReturn = safeRatio(candles[i + options.predictionWindowMinutes].close, now.close) - 1.0;
        sample.tradeReturn = sample.forwardReturn;
        sample.maxFutureHighReturn = safeRatio(futureHigh, now.close) - 1.0;
        sample.maxFutureLowReturn = safeRatio(futureLow, now.close) - 1.0;
        sample.quoteVolume = now.quoteVolume;

        if (options.labelMode == "target_stop") {
            const double targetPrice = now.close * (1.0 + options.upsideTarget);
            const double stopPrice = now.close * (1.0 - options.downsideStop);
            bool skipSample = false;
            sample.label = 0;
            for (int forward = 1; forward <= options.predictionWindowMinutes; ++forward) {
                const bool hitStop = candles[i + forward].low <= stopPrice;
                const bool hitTarget = candles[i + forward].high >= targetPrice;
                // Intraminute ordering is unknown in kline data, so ties must be explicit.
                if (hitStop && hitTarget) {
                    if (options.tiePolicy == "skip") {
                        skipSample = true;
                        break;
                    }
                    if (options.tiePolicy == "target_first") {
                        sample.label = 1;
                        sample.tradeReturn = options.upsideTarget;
                        break;
                    }
                }
                if (hitStop) {
                    sample.label = 0;
                    sample.tradeReturn = -options.downsideStop;
                    break;
                }
                if (hitTarget) {
                    sample.label = 1;
                    sample.tradeReturn = options.upsideTarget;
                    break;
                }
            }
            if (skipSample) {
                continue;
            }
        } else {
            sample.label = sample.maxFutureHighReturn >= options.growthThreshold ? 1 : 0;
            if (sample.label == 1) {
                sample.tradeReturn = options.growthThreshold;
            }
        }

        const double range = std::max(0.0, now.high - now.low);
        const double body = std::fabs(now.close - now.open);
        const double upperWick = std::max(0.0, now.high - std::max(now.open, now.close));
        const double lowerWick = std::max(0.0, std::min(now.open, now.close) - now.low);
        const size_t rollingBegin = i - static_cast<size_t>(kRollingLookbackMinutes);
        double volumeMean = 0.0;
        double volumeStddev = 1.0;
        double tradeMean = 0.0;
        double tradeStddev = 1.0;
        rollingMeanStd(candles, rollingBegin, i, false, volumeMean, volumeStddev);
        rollingMeanStd(candles, rollingBegin, i, true, tradeMean, tradeStddev);
        double recentHigh = 0.0;
        double recentLow = std::numeric_limits<double>::max();
        for (size_t lookback = rollingBegin + 1; lookback <= i; ++lookback) {
            recentHigh = std::max(recentHigh, candles[lookback].high);
            recentLow = std::min(recentLow, candles[lookback].low);
        }

        sample.features.push_back(clipped(safeRatio(now.close, candles[i - 1].close) - 1.0, -1.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.close, candles[i - 3].close) - 1.0, -1.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.close, candles[i - 5].close) - 1.0, -1.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.high - now.low, now.close), 0.0, 2.0));
        sample.features.push_back(clipped(safeRatio(now.close - now.open, now.open), -1.0, 1.0));
        sample.features.push_back(std::log(1.0 + std::max(0.0, now.volume)));
        sample.features.push_back(std::log(1.0 + std::max(0.0, now.quoteVolume)));
        sample.features.push_back(std::log(1.0 + std::max(0.0, now.trades)));
        sample.features.push_back(takerBuyRatio(now));
        sample.features.push_back(clipped(safeRatio(now.volume, candles[i - 1].volume + 1e-12) - 1.0, -10.0, 10.0));
        sample.features.push_back(closeReturn(candles, i, 10));
        sample.features.push_back(closeReturn(candles, i, 15));
        sample.features.push_back(closeReturn(candles, i, 30));
        sample.features.push_back(closeReturn(candles, i, 60));
        sample.features.push_back(clipped(rollingReturnVolatility(candles, i, kRollingLookbackMinutes), 0.0, 1.0));
        sample.features.push_back(std::log(1.0 + std::max(0.0, volumeMean)));
        sample.features.push_back(clipped((now.volume - volumeMean) / volumeStddev, -20.0, 20.0));
        sample.features.push_back(clipped(safeRatio(now.volume, volumeMean + 1e-12), 0.0, 50.0));
        sample.features.push_back(clipped((now.trades - tradeMean) / tradeStddev, -20.0, 20.0));
        sample.features.push_back(clipped(takerBuyRatio(now) - takerBuyRatio(candles[i - 1]), -1.0, 1.0));
        sample.features.push_back(clipped(2.0 * takerBuyRatio(now) - 1.0, -1.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.close, recentHigh) - 1.0, -1.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.close, recentLow) - 1.0, -1.0, 1.0));
        sample.features.push_back(static_cast<double>(consecutiveCandles(candles, i, true)));
        sample.features.push_back(static_cast<double>(consecutiveCandles(candles, i, false)));
        sample.features.push_back(clipped(
            (safeRatio(now.volume, candles[i - 1].volume + 1e-12) - 1.0)
                - (safeRatio(candles[i - 1].volume, candles[i - 2].volume + 1e-12) - 1.0),
            -20.0,
            20.0));
        sample.features.push_back(clipped(
            (safeRatio(now.trades, candles[i - 1].trades + 1e-12) - 1.0)
                - (safeRatio(candles[i - 1].trades, candles[i - 2].trades + 1e-12) - 1.0),
            -20.0,
            20.0));
        sample.features.push_back(clipped(safeRatio(body, range), 0.0, 1.0));
        sample.features.push_back(clipped(safeRatio(upperWick, range), 0.0, 1.0));
        sample.features.push_back(clipped(safeRatio(lowerWick, range), 0.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.close, context.high24h[i]) - 1.0, -1.0, 1.0));
        sample.features.push_back(clipped(safeRatio(now.close, context.low24h[i]) - 1.0, -1.0, 1.0));
        sample.features.push_back(std::log(1.0 + std::max(0.0, context.quoteVolumeMean[i])));
        sample.features.push_back(clipped(
            (now.quoteVolume - context.quoteVolumeMean[i]) / context.quoteVolumeStddev[i],
            -20.0,
            20.0));
        sample.features.push_back(clipped(
            safeRatio(now.quoteVolume, context.quoteVolumeMean[i] + 1e-12),
            0.0,
            50.0));
        sample.features.push_back(clipped(context.takerBuyMean[i], 0.0, 1.0));
        sample.features.push_back(clipped(
            (takerBuyRatio(now) - context.takerBuyMean[i]) / context.takerBuyStddev[i],
            -20.0,
            20.0));

        samples.push_back(sample);
    }

    return samples;
}

const Sample *sampleAtTime(const std::vector<Sample> &samples, long long timeOrder) {
    for (size_t i = 0; i < samples.size(); ++i) {
        if (samples[i].timeOrder == timeOrder) {
            return &samples[i];
        }
    }
    return NULL;
}

void runSelfTests() {
    std::vector<Candle> base(70);
    for (size_t i = 0; i < base.size(); ++i) {
        base[i].openTime = static_cast<long long>(i) * 60000LL;
        base[i].open = 100.0;
        base[i].high = 100.2;
        base[i].low = 99.8;
        base[i].close = 100.0;
        base[i].volume = 1000.0;
        base[i].quoteVolume = 100000.0;
        base[i].trades = 100.0;
        base[i].takerBuyBaseVolume = 500.0;
    }

    ScraperOptions options;
    options.labelMode = "target_stop";
    options.predictionWindowMinutes = 5;
    options.upsideTarget = 0.05;
    options.downsideStop = 0.02;

    std::vector<Candle> target = base;
    target[61].high = 106.0;
    const std::vector<Sample> targetSamples = makeSamples("TEST", "2020-01", 0, target, options);
    const Sample *targetSample = sampleAtTime(targetSamples, 60LL * 60000LL);
    if (!targetSample || targetSample->label != 1 || std::fabs(targetSample->tradeReturn - 0.05) > 1e-12) {
        throw std::runtime_error("self-test failed: target-first outcome");
    }

    std::vector<Candle> stop = base;
    stop[61].low = 97.0;
    const std::vector<Sample> stopSamples = makeSamples("TEST", "2020-01", 0, stop, options);
    const Sample *stopSample = sampleAtTime(stopSamples, 60LL * 60000LL);
    if (!stopSample || stopSample->label != 0 || std::fabs(stopSample->tradeReturn + 0.02) > 1e-12) {
        throw std::runtime_error("self-test failed: stop outcome");
    }

    std::vector<Candle> tie = base;
    tie[61].high = 106.0;
    tie[61].low = 97.0;
    options.tiePolicy = "skip";
    const std::vector<Sample> skippedTieSamples = makeSamples("TEST", "2020-01", 0, tie, options);
    if (sampleAtTime(skippedTieSamples, 60LL * 60000LL)) {
        throw std::runtime_error("self-test failed: skip tie policy");
    }
    options.tiePolicy = "target_first";
    const std::vector<Sample> tieTargetSamples = makeSamples("TEST", "2020-01", 0, tie, options);
    const Sample *tieTarget = sampleAtTime(tieTargetSamples, 60LL * 60000LL);
    if (!tieTarget || tieTarget->label != 1) {
        throw std::runtime_error("self-test failed: target_first tie policy");
    }

    std::cout << "C++ offline self-tests passed.\n";
}

void writeTrainingCsvHeader(std::ostream &out) {
    const std::vector<std::string> names = featureNames();
    out << "symbol,month,month_index,open_time,label,forward_return,trade_return,max_future_high_return,max_future_low_return,quote_volume";
    for (size_t i = 0; i < names.size(); ++i) {
        out << ',' << names[i];
    }
    out << '\n';
    out << std::setprecision(12);
}

void writeTrainingCsvRows(std::ostream &out, const std::vector<Sample> &samples) {
    for (size_t i = 0; i < samples.size(); ++i) {
        out << csvEscape(samples[i].symbol) << ','
            << csvEscape(samples[i].month) << ','
            << samples[i].monthIndex << ','
            << samples[i].timeOrder << ','
            << samples[i].label << ','
            << samples[i].forwardReturn << ','
            << samples[i].tradeReturn << ','
            << samples[i].maxFutureHighReturn << ','
            << samples[i].maxFutureLowReturn << ','
            << samples[i].quoteVolume;
        for (size_t j = 0; j < samples[i].features.size(); ++j) {
            out << ',' << samples[i].features[j];
        }
        out << '\n';
    }
}

class TrainingCsvWriter {
public:
    TrainingCsvWriter() : out_(kTrainingCsv.c_str()), rowsWritten_(0) {
        if (!out_) {
            throw std::runtime_error("Unable to open training CSV for writing");
        }
        writeTrainingCsvHeader(out_);
    }

    void write(const std::vector<Sample> &samples) {
        writeTrainingCsvRows(out_, samples);
        rowsWritten_ += samples.size();
        out_.flush();
    }

    size_t rowsWritten() const {
        return rowsWritten_;
    }

private:
    std::ofstream out_;
    size_t rowsWritten_;
};

void writeTrainingCsv(const std::vector<Sample> &samples) {
    TrainingCsvWriter writer;
    writer.write(samples);
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

std::vector<double> predictProbabilities(
    const std::vector<Sample> &samples,
    const Scaler &scaler,
    const std::vector<double> &weights) {
    std::vector<double> probabilities;
    probabilities.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        probabilities.push_back(predictProbability(samples[i], scaler, weights));
    }
    return probabilities;
}

double medianValue(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

long long minuteBucketForTimestamp(long long timestamp) {
    const long long absolute = timestamp < 0 ? -timestamp : timestamp;
    if (absolute >= 100000000000000LL) {
        return timestamp / (60LL * 1000000LL);
    }
    if (absolute >= 100000000000LL) {
        return timestamp / (60LL * 1000LL);
    }
    return timestamp / 60LL;
}

struct OpenPortfolioPosition {
    long long closeMinute;
    size_t sequence;
    double size;
    double profit;
};

struct OpenPortfolioPositionSooner {
    bool operator()(const OpenPortfolioPosition &left, const OpenPortfolioPosition &right) const {
        if (left.closeMinute != right.closeMinute) {
            return left.closeMinute > right.closeMinute;
        }
        return left.sequence > right.sequence;
    }
};

struct PortfolioExecution {
    PortfolioExecution()
        : endingCapital(0.0),
          portfolioProfit(0.0),
          portfolioReturn(0.0),
          averagePositionSize(0.0),
          medianPositionSize(0.0),
          averageProfitPerTrade(0.0),
          worstTrade(0.0),
          maxCapitalDrawdown(0.0) {}

    std::map<size_t, double> positions;
    double endingCapital;
    double portfolioProfit;
    double portfolioReturn;
    double averagePositionSize;
    double medianPositionSize;
    double averageProfitPerTrade;
    double worstTrade;
    double maxCapitalDrawdown;
};

PortfolioExecution simulatePortfolio(
    const std::vector<Sample> &samples,
    const std::vector<double> &probabilities,
    double threshold,
    const ScraperOptions &options) {
    std::vector<size_t> signals;
    for (size_t i = 0; i < samples.size(); ++i) {
        if (probabilities[i] >= threshold) {
            signals.push_back(i);
        }
    }
    std::sort(signals.begin(), signals.end(), [&](size_t left, size_t right) {
        const long long leftMinute = minuteBucketForTimestamp(samples[left].timeOrder);
        const long long rightMinute = minuteBucketForTimestamp(samples[right].timeOrder);
        if (leftMinute != rightMinute) {
            return leftMinute < rightMinute;
        }
        return probabilities[left] > probabilities[right];
    });

    PortfolioExecution result;
    double cash = options.initialCapital;
    double invested = 0.0;
    double peakEquity = options.initialCapital;
    std::deque<long long> recentEntryMinutes;
    std::priority_queue<OpenPortfolioPosition, std::vector<OpenPortfolioPosition>, OpenPortfolioPositionSooner> openPositions;
    size_t sequence = 0;
    double positionSizeSum = 0.0;
    std::vector<double> positionSizes;
    std::vector<double> positionProfits;

    const double fixedPositionCap = options.initialCapital * options.maxPositionFraction;
    const double feeAndSlippage = options.fee + options.slippage;
    const auto releasePositions = [&](long long untilMinute, double &releaseCash, double &releaseInvested,
                                      double &releasePeak, double &releaseDrawdown) {
        while (!openPositions.empty() && openPositions.top().closeMinute <= untilMinute) {
            const OpenPortfolioPosition position = openPositions.top();
            openPositions.pop();
            releaseInvested -= position.size;
            releaseCash += position.size + position.profit;
            const double equity = releaseCash + releaseInvested;
            releasePeak = std::max(releasePeak, equity);
            releaseDrawdown = std::max(releaseDrawdown, releasePeak - equity);
        }
    };

    for (size_t order = 0; order < signals.size(); ++order) {
        const size_t index = signals[order];
        const long long minute = minuteBucketForTimestamp(samples[index].timeOrder);
        releasePositions(minute, cash, invested, peakEquity, result.maxCapitalDrawdown);
        while (!recentEntryMinutes.empty()
                && recentEntryMinutes.front() <= minute - options.tradePeriodMinutes) {
            recentEntryMinutes.pop_front();
        }
        if (recentEntryMinutes.size() >= static_cast<size_t>(options.maxTradesPerPeriod)) {
            continue;
        }
        const double volumeCap = samples[index].quoteVolume * options.maxVolumeFraction;
        const double positionSize = std::min(fixedPositionCap, std::min(volumeCap, cash));
        if (positionSize <= 0.0) {
            continue;
        }

        OpenPortfolioPosition position;
        position.closeMinute = minute + options.holdingPeriodMinutes;
        position.sequence = sequence++;
        position.size = positionSize;
        position.profit = positionSize * (samples[index].tradeReturn - feeAndSlippage);
        openPositions.push(position);
        cash -= positionSize;
        invested += positionSize;
        recentEntryMinutes.push_back(minute);
        result.positions[index] = positionSize;
        positionSizeSum += positionSize;
        positionSizes.push_back(positionSize);
        positionProfits.push_back(position.profit);
    }

    releasePositions(LLONG_MAX, cash, invested, peakEquity, result.maxCapitalDrawdown);
    result.endingCapital = cash;
    result.portfolioProfit = cash - options.initialCapital;
    result.portfolioReturn = result.portfolioProfit / options.initialCapital;
    result.averagePositionSize = result.positions.empty()
        ? 0.0
        : positionSizeSum / static_cast<double>(result.positions.size());
    result.medianPositionSize = medianValue(positionSizes);
    result.averageProfitPerTrade = result.positions.empty()
        ? 0.0
        : result.portfolioProfit / static_cast<double>(result.positions.size());
    result.worstTrade = positionProfits.empty()
        ? 0.0
        : *std::min_element(positionProfits.begin(), positionProfits.end());
    return result;
}

EvaluationMetrics evaluatePredictions(
    const std::vector<Sample> &samples,
    const std::vector<double> &probabilities,
    double threshold,
    const ScraperOptions &options) {
    EvaluationMetrics metrics;
    metrics.rows = static_cast<int>(samples.size());
    metrics.threshold = threshold;

    std::vector<std::pair<double, int> > scores;
    std::vector<double> forwardReturns;
    std::vector<double> tradeReturns;
    double sumForwardReturn = 0.0;
    double sumTradeReturn = 0.0;
    double sumMfe = 0.0;
    double sumMae = 0.0;
    double grossProfit = 0.0;
    double grossLoss = 0.0;
    double equity = 0.0;
    double peakEquity = 0.0;
    int winningTrades = 0;
    std::set<long long> tradingDays;
    std::set<std::string> tradingMonths;
    const PortfolioExecution execution = simulatePortfolio(samples, probabilities, threshold, options);

    for (size_t i = 0; i < samples.size(); ++i) {
        scores.push_back(std::pair<double, int>(probabilities[i], samples[i].label));
        if (samples[i].label == 1) {
            ++metrics.actualPositiveRows;
        }

        const bool predicted = execution.positions.find(i) != execution.positions.end();
        if (!predicted) {
            if (samples[i].label == 1) {
                ++metrics.falseNegativeRows;
            } else {
                ++metrics.trueNegativeRows;
            }
            continue;
        }

        ++metrics.predictedTrades;
        tradingDays.insert(minuteBucketForTimestamp(samples[i].timeOrder) / (24LL * 60LL));
        tradingMonths.insert(samples[i].month);
        if (samples[i].label == 1) {
            ++metrics.truePositiveRows;
        } else {
            ++metrics.falsePositiveRows;
        }

        const double afterFee = samples[i].tradeReturn - options.fee;
        const double afterFeeAndSlippage = samples[i].tradeReturn - options.fee - options.slippage;
        metrics.totalProfitAfterFee += afterFee;
        metrics.totalProfitAfterFeeAndSlippage += afterFeeAndSlippage;
        sumForwardReturn += samples[i].forwardReturn;
        sumTradeReturn += samples[i].tradeReturn;
        sumMfe += samples[i].maxFutureHighReturn;
        sumMae += samples[i].maxFutureLowReturn;
        forwardReturns.push_back(samples[i].forwardReturn);
        tradeReturns.push_back(samples[i].tradeReturn);
        if (samples[i].tradeReturn > 0.0) {
            ++winningTrades;
        }
        if (afterFeeAndSlippage >= 0.0) {
            grossProfit += afterFeeAndSlippage;
        } else {
            grossLoss += -afterFeeAndSlippage;
        }
        equity += afterFeeAndSlippage;
        peakEquity = std::max(peakEquity, equity);
        metrics.maxDrawdown = std::max(metrics.maxDrawdown, peakEquity - equity);
    }

    metrics.aucScore = auc(scores);
    metrics.accuracy = metrics.rows > 0
        ? static_cast<double>(metrics.truePositiveRows + metrics.trueNegativeRows) / static_cast<double>(metrics.rows)
        : 0.0;
    metrics.precision = metrics.predictedTrades > 0
        ? static_cast<double>(metrics.truePositiveRows) / static_cast<double>(metrics.predictedTrades)
        : 0.0;
    metrics.recall = metrics.actualPositiveRows > 0
        ? static_cast<double>(metrics.truePositiveRows) / static_cast<double>(metrics.actualPositiveRows)
        : 0.0;
    metrics.f1 = (metrics.precision + metrics.recall) > 0.0
        ? 2.0 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        : 0.0;

    if (metrics.predictedTrades > 0) {
        const double tradeCount = static_cast<double>(metrics.predictedTrades);
        metrics.winRate = static_cast<double>(winningTrades) / tradeCount;
        metrics.averageForwardReturn = sumForwardReturn / tradeCount;
        metrics.medianForwardReturn = medianValue(forwardReturns);
        metrics.averageTradeReturn = sumTradeReturn / tradeCount;
        metrics.medianTradeReturn = medianValue(tradeReturns);
        metrics.averageMaxFavorableExcursion = sumMfe / tradeCount;
        metrics.averageMaxAdverseExcursion = sumMae / tradeCount;
        metrics.averageProfitAfterFee = metrics.totalProfitAfterFee / tradeCount;
        metrics.averageProfitAfterFeeAndSlippage = metrics.totalProfitAfterFeeAndSlippage / tradeCount;
        metrics.profitFactor = grossLoss > 0.0
            ? grossProfit / grossLoss
            : (grossProfit > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
    }
    metrics.initialCapital = options.initialCapital;
    metrics.endingCapital = execution.endingCapital;
    metrics.portfolioProfit = execution.portfolioProfit;
    metrics.portfolioReturn = execution.portfolioReturn;
    metrics.averagePositionSize = execution.averagePositionSize;
    metrics.medianPositionSize = execution.medianPositionSize;
    metrics.tradesPerDay = tradingDays.empty()
        ? 0.0
        : static_cast<double>(metrics.predictedTrades) / static_cast<double>(tradingDays.size());
    metrics.tradesPerMonth = tradingMonths.empty()
        ? 0.0
        : static_cast<double>(metrics.predictedTrades) / static_cast<double>(tradingMonths.size());
    metrics.averageProfitPerTrade = execution.averageProfitPerTrade;
    metrics.worstTrade = execution.worstTrade;
    metrics.maxCapitalDrawdown = execution.maxCapitalDrawdown;

    return metrics;
}

void addThresholdCandidate(std::vector<double> &thresholds, double value) {
    if (value < 0.0) {
        value = 0.0;
    } else if (value > 1.0) {
        value = 1.0;
    }
    thresholds.push_back(value);
}

std::vector<double> adaptiveThresholdGrid(
    const std::vector<double> &baseThresholds,
    const std::vector<double> &probabilities,
    int minValidationTrades) {
    std::vector<double> thresholds = baseThresholds;
    std::vector<double> ordered;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        if (probabilities[i] >= 0.0 && probabilities[i] <= 1.0) {
            ordered.push_back(probabilities[i]);
        }
    }
    if (ordered.empty()) {
        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
        return thresholds;
    }

    std::sort(ordered.begin(), ordered.end());
    const double quantiles[] = {0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995};
    for (size_t i = 0; i < sizeof(quantiles) / sizeof(quantiles[0]); ++i) {
        const size_t index = static_cast<size_t>((ordered.size() - 1) * quantiles[i]);
        addThresholdCandidate(thresholds, ordered[index]);
    }

    const int targetCounts[] = {
        std::max(1, minValidationTrades),
        std::max(1, minValidationTrades * 2),
        10,
        25,
        50,
        100,
        250,
        500,
        1000
    };
    for (size_t i = 0; i < sizeof(targetCounts) / sizeof(targetCounts[0]); ++i) {
        const int count = targetCounts[i];
        if (count > 0 && static_cast<size_t>(count) <= ordered.size()) {
            addThresholdCandidate(thresholds, ordered[ordered.size() - static_cast<size_t>(count)]);
        }
    }

    addThresholdCandidate(thresholds, std::max(0.0, ordered.back() - 1e-12));
    std::sort(thresholds.begin(), thresholds.end());
    std::vector<double> uniqueThresholds;
    for (size_t i = 0; i < thresholds.size(); ++i) {
        if (uniqueThresholds.empty() || std::fabs(thresholds[i] - uniqueThresholds.back()) > 1e-12) {
            uniqueThresholds.push_back(thresholds[i]);
        }
    }
    return uniqueThresholds;
}

double thresholdScore(const EvaluationMetrics &metrics, const std::string &objective, double zeroTradeProfitScore) {
    if (metrics.predictedTrades == 0) {
        return objective == "profit" ? zeroTradeProfitScore : -std::numeric_limits<double>::infinity();
    }
    if (objective == "precision") {
        return metrics.precision;
    }
    if (objective == "recall") {
        return metrics.recall;
    }
    if (objective == "f1") {
        return metrics.f1;
    }
    return metrics.portfolioProfit;
}

double tuneThreshold(
    const std::vector<Sample> &validationSamples,
    const std::vector<double> &validationProbabilities,
    const ScraperOptions &options,
    EvaluationMetrics &bestMetrics) {
    const std::vector<double> thresholds = options.adaptiveThresholds
        ? adaptiveThresholdGrid(options.thresholds, validationProbabilities, options.minValidationTrades)
        : options.thresholds;
    double bestThreshold = thresholds.empty() ? 0.5 : thresholds[0];
    const bool strictProfit = options.thresholdObjective == "profit" && options.profitSafety == "strict";
    const double zeroTradeProfitScore = strictProfit ? 0.0 : -std::numeric_limits<double>::infinity();
    double bestScore = strictProfit ? 0.0 : -std::numeric_limits<double>::infinity();
    bool hasBest = false;
    double fallbackThreshold = bestThreshold;
    EvaluationMetrics fallbackMetrics;
    bool hasFallback = false;
    double fallbackScore = -std::numeric_limits<double>::infinity();
    if (strictProfit) {
        bestThreshold = 1.01;
        bestMetrics = evaluatePredictions(validationSamples, validationProbabilities, bestThreshold, options);
        hasBest = true;
    }
    for (size_t i = 0; i < thresholds.size(); ++i) {
        const EvaluationMetrics metrics = evaluatePredictions(
            validationSamples,
            validationProbabilities,
            thresholds[i],
            options);
        const double score = thresholdScore(metrics, options.thresholdObjective, zeroTradeProfitScore);
        if (metrics.predictedTrades < options.minValidationTrades) {
            if (metrics.predictedTrades > 0 && score > fallbackScore) {
                fallbackThreshold = thresholds[i];
                fallbackMetrics = metrics;
                fallbackScore = score;
                hasFallback = true;
            }
            continue;
        }
        if (!hasBest || score > bestScore) {
            bestScore = score;
            bestThreshold = thresholds[i];
            bestMetrics = metrics;
            hasBest = true;
        }
    }

    if (!hasBest) {
        if (hasFallback) {
            bestThreshold = fallbackThreshold;
            bestMetrics = fallbackMetrics;
            return bestThreshold;
        }
        bestThreshold = 1.01;
        bestMetrics = evaluatePredictions(validationSamples, validationProbabilities, bestThreshold, options);
    }
    return bestThreshold;
}

void writePredictionsCsv(
    const std::string &path,
    const std::vector<Sample> &samples,
    const std::vector<double> &probabilities,
    double threshold,
    const std::string &modelName,
    const ScraperOptions &options) {
    std::ofstream out(path.c_str());
    if (!out) {
        throw std::runtime_error("Unable to open predictions CSV for writing: " + path);
    }

    out << "symbol,month,month_index,open_time,label,probability,selected_threshold,raw_signal,predicted,position_size,"
        << "forward_return,trade_return,max_future_high_return,max_future_low_return,model_name\n";
    out << std::setprecision(12);
    const PortfolioExecution execution = simulatePortfolio(samples, probabilities, threshold, options);
    for (size_t i = 0; i < samples.size(); ++i) {
        const int rawSignal = probabilities[i] >= threshold ? 1 : 0;
        const std::map<size_t, double>::const_iterator position = execution.positions.find(i);
        const int predicted = position == execution.positions.end() ? 0 : 1;
        const double positionSize = predicted ? position->second : 0.0;
        out << csvEscape(samples[i].symbol) << ','
            << csvEscape(samples[i].month) << ','
            << samples[i].monthIndex << ','
            << samples[i].timeOrder << ','
            << samples[i].label << ','
            << probabilities[i] << ','
            << threshold << ','
            << rawSignal << ','
            << predicted << ','
            << positionSize << ','
            << samples[i].forwardReturn << ','
            << samples[i].tradeReturn << ','
            << samples[i].maxFutureHighReturn << ','
            << samples[i].maxFutureLowReturn << ','
            << csvEscape(modelName) << '\n';
    }
}

TrainingResult trainModel(
    std::vector<Sample> trainSamples,
    std::vector<Sample> validationSamples,
    std::vector<Sample> testSamples,
    const ScraperOptions &options,
    EvaluationMetrics &validationMetrics,
    EvaluationMetrics &testMetrics) {
    if (trainSamples.empty()) {
        throw std::runtime_error("No training samples were created");
    }
    if (validationSamples.empty()) {
        throw std::runtime_error("No validation samples were created");
    }
    if (testSamples.empty()) {
        throw std::runtime_error("No out-of-sample test samples were created");
    }

    std::sort(trainSamples.begin(), trainSamples.end(), [](const Sample &left, const Sample &right) {
        if (left.timeOrder == right.timeOrder) {
            return left.symbol < right.symbol;
        }
        return left.timeOrder < right.timeOrder;
    });
    std::sort(validationSamples.begin(), validationSamples.end(), [](const Sample &left, const Sample &right) {
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
    result.validationRows = static_cast<int>(validationSamples.size());
    result.testRows = static_cast<int>(testSamples.size());
    result.positiveRows = 0;

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
    for (size_t i = 0; i < validationSamples.size(); ++i) {
        if (validationSamples[i].label == 1) {
            ++result.positiveRows;
        }
    }
    for (size_t i = 0; i < testSamples.size(); ++i) {
        if (testSamples[i].label == 1) {
            ++result.positiveRows;
        }
    }

    const double positiveWeight = trainPositives > 0
        ? std::min(options.positiveWeightCap, static_cast<double>(trainNegatives) / static_cast<double>(trainPositives))
        : 1.0;

    for (int epoch = 0; epoch < options.epochs; ++epoch) {
        for (size_t i = 0; i < trainSamples.size(); ++i) {
            const double probability = predictProbability(trainSamples[i], result.scaler, result.weights);
            const double classWeight = trainSamples[i].label == 1 ? positiveWeight : 1.0;
            const double error = (probability - static_cast<double>(trainSamples[i].label)) * classWeight;

            result.weights[0] -= options.learningRate * error;
            for (size_t j = 0; j < featureCount; ++j) {
                const double scaled = (trainSamples[i].features[j] - result.scaler.mean[j]) / result.scaler.stddev[j];
                const double gradient = error * scaled + options.l2Regularization * result.weights[j + 1];
                result.weights[j + 1] -= options.learningRate * gradient;
            }
        }
    }

    std::vector<std::pair<double, int> > trainScores;
    for (size_t i = 0; i < trainSamples.size(); ++i) {
        const double probability = predictProbability(trainSamples[i], result.scaler, result.weights);
        trainScores.push_back(std::pair<double, int>(probability, trainSamples[i].label));
    }

    const std::vector<double> validationProbabilities = predictProbabilities(validationSamples, result.scaler, result.weights);
    const std::vector<double> testProbabilities = predictProbabilities(testSamples, result.scaler, result.weights);
    result.trainAuc = auc(trainScores);
    result.selectedThreshold = tuneThreshold(validationSamples, validationProbabilities, options, validationMetrics);
    testMetrics = evaluatePredictions(testSamples, testProbabilities, result.selectedThreshold, options);

    writePredictionsCsv(kLogisticPredictionsCsv, testSamples, testProbabilities, result.selectedThreshold, "logistic", options);
    writePredictionsCsv(kPredictionsCsv, testSamples, testProbabilities, result.selectedThreshold, "logistic", options);
    return result;
}

void writeMetricsCsv(
    const std::string &path,
    const TrainingResult &result,
    const EvaluationMetrics &validationMetrics,
    const EvaluationMetrics &testMetrics,
    const ScraperOptions &options) {
    std::ofstream metrics(path.c_str());
    if (!metrics) {
        throw std::runtime_error("Unable to open metrics CSV for writing: " + path);
    }

    metrics << "model,threshold_objective,selected_threshold,train_rows,validation_rows,test_rows,positive_rows,"
        << "train_auc,validation_auc,test_auc,test_accuracy,test_precision,test_recall,test_f1,"
        << "predicted_trades,true_positive_rows,false_positive_rows,win_rate,"
        << "average_forward_return,median_forward_return,average_trade_return,median_trade_return,"
        << "average_max_favorable_excursion,average_max_adverse_excursion,"
        << "average_profit_after_fee,average_profit_after_fee_and_slippage,total_profit_after_fee,"
        << "total_profit_after_fee_and_slippage,profit_factor,max_drawdown,"
        << "initial_capital,ending_capital,portfolio_profit,portfolio_return,average_position_size,median_position_size,"
        << "trades_per_day,trades_per_month,average_profit_per_trade,worst_trade,max_capital_drawdown,"
        << "fee,slippage,max_position_fraction,max_volume_fraction,max_trades_per_period,trade_period_minutes,holding_period_minutes,"
        << "min_validation_trades,profit_safety,adaptive_thresholds\n";
    metrics << "logistic,"
            << csvEscape(options.thresholdObjective) << ','
            << result.selectedThreshold << ','
            << result.trainRows << ','
            << result.validationRows << ','
            << result.testRows << ','
            << result.positiveRows << ','
            << std::fixed << std::setprecision(8)
            << result.trainAuc << ','
            << validationMetrics.aucScore << ','
            << testMetrics.aucScore << ','
            << testMetrics.accuracy << ','
            << testMetrics.precision << ','
            << testMetrics.recall << ','
            << testMetrics.f1 << ','
            << testMetrics.predictedTrades << ','
            << testMetrics.truePositiveRows << ','
            << testMetrics.falsePositiveRows << ','
            << testMetrics.winRate << ','
            << testMetrics.averageForwardReturn << ','
            << testMetrics.medianForwardReturn << ','
            << testMetrics.averageTradeReturn << ','
            << testMetrics.medianTradeReturn << ','
            << testMetrics.averageMaxFavorableExcursion << ','
            << testMetrics.averageMaxAdverseExcursion << ','
            << testMetrics.averageProfitAfterFee << ','
            << testMetrics.averageProfitAfterFeeAndSlippage << ','
            << testMetrics.totalProfitAfterFee << ','
            << testMetrics.totalProfitAfterFeeAndSlippage << ','
            << testMetrics.profitFactor << ','
            << testMetrics.maxDrawdown << ','
            << testMetrics.initialCapital << ','
            << testMetrics.endingCapital << ','
            << testMetrics.portfolioProfit << ','
            << testMetrics.portfolioReturn << ','
            << testMetrics.averagePositionSize << ','
            << testMetrics.medianPositionSize << ','
            << testMetrics.tradesPerDay << ','
            << testMetrics.tradesPerMonth << ','
            << testMetrics.averageProfitPerTrade << ','
            << testMetrics.worstTrade << ','
            << testMetrics.maxCapitalDrawdown << ','
            << options.fee << ','
            << options.slippage << ','
            << options.maxPositionFraction << ','
            << options.maxVolumeFraction << ','
            << options.maxTradesPerPeriod << ','
            << options.tradePeriodMinutes << ','
            << options.holdingPeriodMinutes << ','
            << options.minValidationTrades << ','
            << csvEscape(options.profitSafety) << ','
            << (options.adaptiveThresholds ? 1 : 0) << '\n';
}

void writeModel(
    const TrainingResult &result,
    const EvaluationMetrics &validationMetrics,
    const EvaluationMetrics &testMetrics,
    const ScraperOptions &options) {
    const std::vector<std::string> names = featureNames();
    std::ofstream model(kModelCsv.c_str());
    if (!model) {
        throw std::runtime_error("Unable to open model CSV for writing");
    }

    model << "feature,weight,mean,stddev\n";
    model << std::setprecision(12);
    model << "intercept," << result.weights[0] << ",0,1\n";
    for (size_t i = 0; i < result.scaler.mean.size(); ++i) {
        model << names[i] << ','
              << result.weights[i + 1] << ','
              << result.scaler.mean[i] << ','
              << result.scaler.stddev[i] << '\n';
    }

    writeMetricsCsv(kLogisticMetricsCsv, result, validationMetrics, testMetrics, options);
    writeMetricsCsv(kMetricsCsv, result, validationMetrics, testMetrics, options);
}

} // namespace

void scrapeHistoricalCoinData(const std::vector<std::string> &symbolOverrides) {
    try {
        ScraperOptions options;
        std::vector<std::string> symbolArgs;
        if (!parseArguments(symbolOverrides, options, symbolArgs)) {
            return;
        }
        if (options.selfTest) {
            runSelfTests();
            return;
        }
        const std::vector<std::string> symbols = readRequestedSymbols(symbolArgs);
        std::vector<Sample> trainingSamples;
        std::vector<Sample> validationSamples;
        std::vector<Sample> testSamples;
        TrainingCsvWriter trainingWriter;

        for (size_t i = 0; i < symbols.size(); ++i) {
            try {
                const std::vector<std::string> dates = listKlineDates(symbols[i]);
                const std::vector<std::string> months = firstAvailableMonths(dates, options.totalMonths);
                const MonthSplit split = splitForMonthCount(static_cast<int>(months.size()), options);
                const int requiredMonths = split.train + split.validation + split.test;
                if (static_cast<int>(months.size()) < requiredMonths || split.train <= 0 || split.validation <= 0 || split.test <= 0) {
                    std::cout << "Skipping " << symbols[i] << ": fewer than "
                              << (options.splitMode == "fixed" ? options.requiredMonths() : 3)
                              << " months of 1m klines.\n";
                    continue;
                }

                size_t symbolTrainSamples = 0;
                size_t symbolValidationSamples = 0;
                size_t symbolTestSamples = 0;
                for (size_t monthIndex = 0; monthIndex < months.size(); ++monthIndex) {
                    const std::vector<Candle> candles = downloadCandlesForMonth(
                        symbols[i],
                        months[monthIndex],
                        dates);
                    const std::vector<Sample> monthSamples = makeSamples(
                        symbols[i],
                        months[monthIndex],
                        static_cast<int>(monthIndex),
                        candles,
                        options);
                    trainingWriter.write(monthSamples);
                    std::cout << "Generated symbol=" << symbols[i]
                              << " month=" << months[monthIndex]
                              << " candles_loaded=" << candles.size()
                              << " rows_written=" << monthSamples.size()
                              << " cumulative_rows=" << trainingWriter.rowsWritten()
                              << '\n' << std::flush;

                    if (static_cast<int>(monthIndex) < split.train) {
                        symbolTrainSamples += monthSamples.size();
                        if (!options.generateOnly) {
                            trainingSamples.insert(trainingSamples.end(), monthSamples.begin(), monthSamples.end());
                        }
                    } else if (static_cast<int>(monthIndex) < split.train + split.validation) {
                        symbolValidationSamples += monthSamples.size();
                        if (!options.generateOnly) {
                            validationSamples.insert(validationSamples.end(), monthSamples.begin(), monthSamples.end());
                        }
                    } else if (static_cast<int>(monthIndex) < split.train + split.validation + split.test) {
                        symbolTestSamples += monthSamples.size();
                        if (!options.generateOnly) {
                            testSamples.insert(testSamples.end(), monthSamples.begin(), monthSamples.end());
                        }
                    }
                }

                std::cout << "Processed " << (i + 1) << "/" << symbols.size()
                          << ": " << symbols[i]
                          << " split_mode=" << options.splitMode
                          << " available_months=" << months.size()
                          << " train_months=" << monthRangeDescription(months, 0, split.train)
                          << " train_samples=" << symbolTrainSamples
                          << " validation_months=" << monthRangeDescription(months, split.train, split.validation)
                          << " validation_samples=" << symbolValidationSamples
                          << " test_months=" << monthRangeDescription(months, split.train + split.validation, split.test)
                          << " test_samples=" << symbolTestSamples << '\n';
            } catch (const std::exception &error) {
                std::cerr << "\nSkipping " << symbols[i] << ": " << error.what() << '\n';
                continue;
            }
        }

        if (options.generateOnly) {
            std::cout << "Wrote " << kTrainingCsv
                      << ". Skipped C++ logistic baseline because --generate-only was set.\n";
            return;
        }

        EvaluationMetrics validationMetrics;
        EvaluationMetrics testMetrics;
        const TrainingResult result = trainModel(trainingSamples, validationSamples, testSamples, options, validationMetrics, testMetrics);
        writeModel(result, validationMetrics, testMetrics, options);
        const MonthSevenEvaluation evaluation = evaluateMonthSevenPredictions(kPredictionsCsv);

        std::cout << "Wrote " << kTrainingCsv << ", " << kLogisticPredictionsCsv << ", "
                  << kModelCsv << ", and " << kLogisticMetricsCsv << ".\n"
                  << "Train AUC: " << std::fixed << std::setprecision(4) << result.trainAuc
                  << " | Validation threshold: " << result.selectedThreshold
                  << " | Test AUC: " << testMetrics.aucScore
                  << " | Success rate: " << (evaluation.successRate * 100.0) << "%"
                  << " | Accuracy: " << (testMetrics.accuracy * 100.0) << "%"
                  << " | Recall: " << (testMetrics.recall * 100.0) << "%"
                  << " | Portfolio profit: " << testMetrics.portfolioProfit
                  << " | Portfolio return: " << (testMetrics.portfolioReturn * 100.0) << "%\n";
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
