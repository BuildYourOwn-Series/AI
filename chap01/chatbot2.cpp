#include <iostream>
#include <string>
#include <cctype>

// Helper: normalize the input by lowercasing and removing punctuation.
std::string normalize(const std::string& text) {
    std::string out;
    for (char c : text) {
        if (std::isalpha(static_cast<unsigned char>(c)) ||
            std::isspace(static_cast<unsigned char>(c))) {
            out += static_cast<char>(
                std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return out;
}

int main() {
    std::string input;
    std::cout << "Ask me something: ";

    while (std::getline(std::cin, input)) {
        std::string norm = normalize(input);

        if (norm == "hello" || norm == "hi" || norm == "greetings") {
            std::cout << "Hello, human." << std::endl;
        } else if (norm == "how are you" || norm == "how are you doing") {
            std::cout << "Operational. You?" << std::endl;
        } else if (norm == "bye" || norm == "goodbye" || norm == "farewell") {
            std::cout << "Goodbye." << std::endl;
            break;
        } else {
            std::cout << "I don't understand.\n";
        }

        std::cout << "\nAsk me something: ";
    }
}
