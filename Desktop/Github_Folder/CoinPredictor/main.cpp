#include <iostream>
#include <string>
#include <vector>
using namespace std;

void scrapeHistoricalCoinData();
void scrapeHistoricalCoinData(const vector<string> &symbolOverrides);

int main(int argc, char **argv) {
    //The objective is to make an algorithm that is able to select and buy coins that
    //are likely to rapidly increase in value
    
    cout << "Welcome to CoinPredictor!" << endl;
    if (argc > 1 && string(argv[1]) == "train") {
        vector<string> symbols;
        for (int i = 2; i < argc; ++i) {
            symbols.push_back(argv[i]);
        }
        scrapeHistoricalCoinData(symbols);
    } else {
        cout << "Run `coin_predictor train` to train from CSV symbols, or `coin_predictor train BTCUSDT` for a smaller run." << endl;
    }

    return 0;
}
