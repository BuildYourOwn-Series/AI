#include <iostream>
#include <string>

int main() {
    std::string input;
    std::cout << "Ask me something: ";
    while (std::getline(std::cin, input)) {
        if (input == "hello") {
            std::cout << "Hello, human." << std::endl;
        } else if (input == "how are you") {
            std::cout << "Operational. You?" << std::endl;
        } else if (input == "bye") {
            std::cout << "Goodbye." << std::endl;
            break;
        } else {
            std::cout << "I don't understand.\n";
        }
        std::cout << "\nAsk me something: ";
    }
}
