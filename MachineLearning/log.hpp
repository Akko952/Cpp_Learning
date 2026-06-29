#ifndef LOG_HPP
#define LOG_HPP

#include <fstream>
#include <iostream>
#include <string>

namespace Logger {
inline void log(const std::string &message,
		const std::string &filename = "log.txt") {
    std::ofstream logFile(filename, std::ios::app);
    if (logFile.is_open()) {
	logFile << message << std::endl;
	logFile.close();
    } else {
	std::cerr << "Unable to open log file." << std::endl;
    }
}
} // namespace Logger
#endif // LOG_HPP