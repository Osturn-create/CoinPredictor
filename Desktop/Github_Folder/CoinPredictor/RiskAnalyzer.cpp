// RiskAnalyzer scrapes Binance Data Vision daily spot trade files and creates
// a CSV of symbols whose 3-day buy/sell trade ratio is between 0.7 and 1.8.
//
// Build standalone:
//   g++ -DRISK_ANALYZER_STANDALONE RiskAnalyzer.cpp -o risk_analyzer
//
// Update the CSV:
//   ./risk_analyzer
//
// Optional: pass specific Binance symbols to limit the scan:
//   ./risk_analyzer BTCUSDT ETHUSDT SOLUSDT

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "RiskAnalyzer.h"

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

namespace {

const std::string kDataVisionBase = "https://data.binance.vision";
const std::string kDataVisionBucket = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision";
const std::string kOutputCsv = "binance_buy_sell_ratio.csv";
const double kMinimumBuySellRatio = 0.70;
const double kMaximumBuySellRatio = 1.80;
const int kDaysToAnalyze = 3;
const char *const kExcludedUsdtBaseAssets[] = {
    "BUSD", "FDUSD", "TUSD", "USDC", "USDP", "USDS", "DAI", "AEUR", "EURI",
    "EUR", "TRY", "BRL", "GBP", "AUD", "JPY", "MXN", "PLN", "RON", "RUB",
    "UAH", "ZAR", "IDR", "BIDR", "NGN", "ARS", "COP", "CZK", "KZT", "VAI",
    "UST", "USTC", "RLUSD", "XUSD", "BFUSD", "SUSD", "USDE", "USDSB", "USD1",
    "PAX", "BKRW"
};
const size_t kExcludedUsdtBaseAssetCount = sizeof(kExcludedUsdtBaseAssets) / sizeof(kExcludedUsdtBaseAssets[0]);

struct SymbolRatio {
    SymbolRatio()
        : buyTrades(0),
          sellTrades(0),
          totalTrades(0),
          buySellRatio(0.0) {}

    std::string symbol;
    long long buyTrades;
    long long sellTrades;
    long long totalTrades;
    double buySellRatio;
    std::vector<std::string> datesUsed;
};

struct SymbolRatioTotalTradesDescending {
    bool operator()(const SymbolRatio &left, const SymbolRatio &right) const {
        return left.totalTrades > right.totalTrades;
    }
};

std::string shellQuote(const std::string &text) {
    std::string quoted = "'";
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(c);
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

bool endsWith(const std::string &text, const std::string &suffix) {
    return text.size() >= suffix.size()
        && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string usdtBaseAsset(const std::string &symbol) {
    return endsWith(symbol, "USDT") ? symbol.substr(0, symbol.size() - 4) : "";
}

bool isExcludedUsdtBaseAsset(const std::string &baseAsset) {
    for (size_t i = 0; i < kExcludedUsdtBaseAssetCount; ++i) {
        if (baseAsset == kExcludedUsdtBaseAssets[i]) {
            return true;
        }
    }
    return false;
}

bool isNormalCryptoUsdtSymbol(const std::string &symbol) {
    const std::string baseAsset = usdtBaseAsset(symbol);
    if (baseAsset.empty() || isExcludedUsdtBaseAsset(baseAsset)) {
        return false;
    }

    const char *const blockedFragments[] = {
        "UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S"
    };
    const size_t blockedFragmentCount = sizeof(blockedFragments) / sizeof(blockedFragments[0]);
    for (size_t i = 0; i < blockedFragmentCount; ++i) {
        const std::string fragment = blockedFragments[i];
        if (baseAsset.size() > fragment.size()
            && baseAsset.find(fragment) != std::string::npos
            && endsWith(baseAsset, fragment)) {
            return false;
        }
    }

    const char *const blockedExactBases[] = {
        "AUSDT", "CUSDT", "DUSDT", "FUSDT", "GUSDT", "SUSDT", "TUSDT",
        "UUSDT", "WUSDT", "WBETH", "WBTC", "BETH", "XAUT"
    };
    const size_t blockedExactBaseCount = sizeof(blockedExactBases) / sizeof(blockedExactBases[0]);
    for (size_t i = 0; i < blockedExactBaseCount; ++i) {
        if (baseAsset == blockedExactBases[i]) {
            return false;
        }
    }

    for (size_t i = 0; i < baseAsset.size(); ++i) {
        const char c = baseAsset[i];
        if (!std::isalnum(static_cast<unsigned char>(c))) {
            return false;
        }
    }

    return true;
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

std::vector<std::string> parseSymbolsFromDataVisionListing(const std::string &listing) {
    std::set<std::string> symbols;
    const std::string prefix = "data/spot/daily/trades/";
    size_t pos = 0;

    while ((pos = listing.find(prefix, pos)) != std::string::npos) {
        const size_t symbolStart = pos + prefix.size();
        const size_t symbolEnd = listing.find('/', symbolStart);
        if (symbolEnd == std::string::npos) {
            break;
        }

        const std::string symbol = listing.substr(symbolStart, symbolEnd - symbolStart);
        if (!symbol.empty()
            && symbol.find('<') == std::string::npos
            && symbol.find('&') == std::string::npos
            && isNormalCryptoUsdtSymbol(symbol)) {
            symbols.insert(symbol);
        }

        pos = symbolEnd + 1;
    }

    return std::vector<std::string>(symbols.begin(), symbols.end());
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

std::vector<std::string> fetchAllBinanceTradeSymbols() {
    std::set<std::string> allSymbols;
    std::string marker;

    while (true) {
        std::string url = kDataVisionBucket + "?delimiter=/&prefix=data/spot/daily/trades/";
        if (!marker.empty()) {
            url += "&marker=" + marker;
        }

        const std::string listing = fetchUrl(url);
        const std::vector<std::string> pageSymbols = parseSymbolsFromDataVisionListing(listing);
        allSymbols.insert(pageSymbols.begin(), pageSymbols.end());

        const std::string nextMarker = parseTagValue(listing, "NextMarker");
        if (nextMarker.empty() || nextMarker == marker) {
            break;
        }
        marker = nextMarker;
    }

    const std::vector<std::string> symbols(allSymbols.begin(), allSymbols.end());
    if (symbols.empty()) {
        throw std::runtime_error("No Binance Data Vision trade symbols found");
    }
    return symbols;
}

std::vector<std::string> lastCompletedUtcDates(int count) {
    std::vector<std::string> dates;
    std::time_t now = std::time(NULL);

    for (int daysBack = 1; static_cast<int>(dates.size()) < count; ++daysBack) {
        const std::time_t day = now - static_cast<std::time_t>(daysBack) * 24 * 60 * 60;
        const std::tm *utc = std::gmtime(&day);
        if (!utc) {
            throw std::runtime_error("Unable to calculate UTC date");
        }

        char date[16];
        if (std::strftime(date, sizeof(date), "%Y-%m-%d", utc) == 0) {
            throw std::runtime_error("Unable to format UTC date");
        }
        dates.push_back(date);
    }

    return dates;
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

std::string tempZipPath(const std::string &symbol, const std::string &date) {
    return "/tmp/binance-trades-" + symbol + "-" + date + ".zip";
}

bool looksLikeHeader(const std::vector<std::string> &fields) {
    return !fields.empty() && fields[0].find_first_not_of("0123456789") != std::string::npos;
}

bool addTradesFromZip(const std::string &zipPath, long long &buyTrades, long long &sellTrades) {
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
    bool foundTrade = false;

    while (std::getline(lines, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') {
            line.erase(line.size() - 1);
        }
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> fields = splitCsvLine(line);
        if (fields.size() < 6 || looksLikeHeader(fields)) {
            continue;
        }

        const std::string isBuyerMaker = fields.size() >= 6 ? fields[5] : "";

        // Binance trade files use isBuyerMaker. false means the buyer was the
        // taker, so the trade was buyer-initiated. true means seller-initiated.
        if (isBuyerMaker == "false" || isBuyerMaker == "False" || isBuyerMaker == "0") {
            ++buyTrades;
            foundTrade = true;
        } else if (isBuyerMaker == "true" || isBuyerMaker == "True" || isBuyerMaker == "1") {
            ++sellTrades;
            foundTrade = true;
        }
    }

    return foundTrade;
}

std::string binanceTradeZipUrl(const std::string &symbol, const std::string &date) {
    return kDataVisionBase + "/data/spot/daily/trades/" + symbol + "/" + symbol + "-trades-" + date + ".zip";
}

bool analyzeSymbol(const std::string &symbol, const std::vector<std::string> &dates, SymbolRatio &result) {
    result.symbol = symbol;

    for (size_t i = 0; i < dates.size(); ++i) {
        const std::string &date = dates[i];
        const std::string path = tempZipPath(symbol, date);
        const std::string url = binanceTradeZipUrl(symbol, date);

        if (!downloadFile(url, path)) {
            if (date == dates.front()) {
                return false;
            }
            continue;
        }

        const long long buysBefore = result.buyTrades;
        const long long sellsBefore = result.sellTrades;
        if (addTradesFromZip(path, result.buyTrades, result.sellTrades)) {
            result.datesUsed.push_back(date);
        } else {
            result.buyTrades = buysBefore;
            result.sellTrades = sellsBefore;
        }

        std::remove(path.c_str());
    }

    result.totalTrades = result.buyTrades + result.sellTrades;
    if (result.sellTrades == 0) {
        result.buySellRatio = result.buyTrades > 0 ? 999999.0 : 0.0;
    } else {
        result.buySellRatio = static_cast<double>(result.buyTrades) / static_cast<double>(result.sellTrades);
    }

    return result.totalTrades > 0
        && result.buySellRatio >= kMinimumBuySellRatio
        && result.buySellRatio <= kMaximumBuySellRatio;
}

std::string joinDates(const std::vector<std::string> &dates) {
    std::ostringstream out;
    for (size_t i = 0; i < dates.size(); ++i) {
        if (i > 0) {
            out << '|';
        }
        out << dates[i];
    }
    return out.str();
}

std::string csvEscape(const std::string &value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";
    for (size_t i = 0; i < value.size(); ++i) {
        const char c = value[i];
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped.push_back(c);
        }
    }
    escaped += '"';
    return escaped;
}

void writeBinanceRatioCsv(const std::vector<SymbolRatio> &ratios, const std::string &path) {
    std::ofstream out(path.c_str());
    if (!out) {
        throw std::runtime_error("Unable to open CSV for writing: " + path);
    }

    out << "symbol,buy_trades_3d,sell_trades_3d,total_trades_3d,buy_sell_ratio,dates_used\n";
    out << std::fixed << std::setprecision(4);

    for (size_t i = 0; i < ratios.size(); ++i) {
        const SymbolRatio &ratio = ratios[i];
        out << csvEscape(ratio.symbol) << ','
            << ratio.buyTrades << ','
            << ratio.sellTrades << ','
            << ratio.totalTrades << ','
            << ratio.buySellRatio << ','
            << csvEscape(joinDates(ratio.datesUsed)) << '\n';
    }
}

std::vector<SymbolRatio> analyzeBinanceDataVision(const std::vector<std::string> &requestedSymbols) {
    const std::vector<std::string> dates = lastCompletedUtcDates(kDaysToAnalyze);
    std::vector<std::string> symbols = requestedSymbols.empty()
        ? fetchAllBinanceTradeSymbols()
        : requestedSymbols;

    std::vector<std::string> filteredSymbols;
    for (size_t i = 0; i < symbols.size(); ++i) {
        if (isNormalCryptoUsdtSymbol(symbols[i])) {
            filteredSymbols.push_back(symbols[i]);
        }
    }
    symbols.swap(filteredSymbols);

    std::vector<SymbolRatio> accepted;
    for (size_t i = 0; i < symbols.size(); ++i) {
        SymbolRatio ratio;
        if (analyzeSymbol(symbols[i], dates, ratio)) {
            accepted.push_back(ratio);
        }

        std::cout << "Checked " << (i + 1) << "/" << symbols.size()
                  << ": " << symbols[i] << '\r' << std::flush;
    }
    std::cout << std::string(80, ' ') << '\r';

    std::sort(accepted.begin(), accepted.end(), SymbolRatioTotalTradesDescending());

    return accepted;
}

} // namespace

int runRiskAnalyzer() {
    return runRiskAnalyzer(std::vector<std::string>());
}

int runRiskAnalyzer(const std::vector<std::string> &symbols) {
    try {
        const std::vector<SymbolRatio> ratios = analyzeBinanceDataVision(symbols);
        writeBinanceRatioCsv(ratios, kOutputCsv);

        const std::time_t now = std::time(NULL);
        char timestamp[32];
        std::tm *local = std::localtime(&now);
        if (local && std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", local) != 0) {
            std::cout << "Updated " << kOutputCsv << " with " << ratios.size()
                      << " Binance symbols at " << timestamp << ".\n";
        } else {
            std::cout << "Updated " << kOutputCsv << " with " << ratios.size()
                      << " Binance symbols.\n";
        }
    } catch (const std::exception &error) {
        std::cerr << "RiskAnalyzer update failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}

#ifdef RISK_ANALYZER_STANDALONE
int main(int argc, char **argv) {
    std::vector<std::string> symbols;
    for (int i = 1; i < argc; ++i) {
        symbols.push_back(argv[i]);
    }

    return runRiskAnalyzer(symbols);
}
#endif
