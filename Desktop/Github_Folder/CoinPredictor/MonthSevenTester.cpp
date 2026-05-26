#include "MonthSevenTester.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

int columnIndex(const std::vector<std::string> &header, const std::string &name, int fallback) {
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) {
            return static_cast<int>(i);
        }
    }
    return fallback;
}

} // namespace

MonthSevenEvaluation evaluateMonthSevenPredictions(const std::string &predictionCsvPath) {
    std::ifstream in(predictionCsvPath.c_str());
    if (!in) {
        throw std::runtime_error("Unable to open 7th-month predictions CSV: " + predictionCsvPath);
    }

    MonthSevenEvaluation result;
    std::string line;
    bool firstLine = true;
    int labelColumn = 3;
    int predictedColumn = 5;
    int trueNegativeRows = 0;
    int falsePositiveRows = 0;
    int falseNegativeRows = 0;

    while (std::getline(in, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') {
            line.erase(line.size() - 1);
        }
        if (line.empty()) {
            continue;
        }
        if (firstLine) {
            firstLine = false;
            const std::vector<std::string> header = splitCsvLine(line);
            labelColumn = columnIndex(header, "label", labelColumn);
            predictedColumn = columnIndex(header, "predicted", predictedColumn);
            continue;
        }

        const std::vector<std::string> fields = splitCsvLine(line);
        if (labelColumn >= static_cast<int>(fields.size())
                || predictedColumn >= static_cast<int>(fields.size())) {
            continue;
        }

        const int label = fields[labelColumn] == "1" ? 1 : 0;
        const int predicted = fields[predictedColumn] == "1" ? 1 : 0;
        ++result.rows;
        if (label == 1) {
            ++result.actualPositiveRows;
        }
        if (predicted == 1) {
            ++result.predictedPositiveRows;
        }

        if (predicted == 1 && label == 1) {
            ++result.truePositiveRows;
        } else if (predicted == 1 && label == 0) {
            ++falsePositiveRows;
        } else if (predicted == 0 && label == 1) {
            ++falseNegativeRows;
        } else {
            ++trueNegativeRows;
        }
    }

    result.successRate = result.predictedPositiveRows > 0
        ? static_cast<double>(result.truePositiveRows) / static_cast<double>(result.predictedPositiveRows)
        : 0.0;
    result.accuracy = result.rows > 0
        ? static_cast<double>(result.truePositiveRows + trueNegativeRows) / static_cast<double>(result.rows)
        : 0.0;
    result.recall = result.actualPositiveRows > 0
        ? static_cast<double>(result.truePositiveRows) / static_cast<double>(result.actualPositiveRows)
        : 0.0;

    (void)falsePositiveRows;
    (void)falseNegativeRows;
    return result;
}
