#ifndef MONTH_SEVEN_TESTER_H
#define MONTH_SEVEN_TESTER_H

#include <string>

struct MonthSevenEvaluation {
    MonthSevenEvaluation()
        : rows(0),
          predictedPositiveRows(0),
          truePositiveRows(0),
          actualPositiveRows(0),
          successRate(0.0),
          accuracy(0.0),
          recall(0.0) {}

    int rows;
    int predictedPositiveRows;
    int truePositiveRows;
    int actualPositiveRows;
    double successRate;
    double accuracy;
    double recall;
};

MonthSevenEvaluation evaluateMonthSevenPredictions(const std::string &predictionCsvPath);

#endif
