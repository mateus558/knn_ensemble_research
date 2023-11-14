#pragma once

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

std::string removeExtension(const std::string& filename) {
    // Find the last dot in the filename
    size_t lastDotPos = filename.find_last_of(".");
    
    // Check if a dot was found and it is not the first character
    if (lastDotPos != std::string::npos && lastDotPos != 0) {
        return filename.substr(0, lastDotPos);
    }
    
    // If no dot or dot is the first character, return the original filename
    return filename;
}

bool createFolder(const std::string& folderName) {
    if (fs::exists(folderName)) {
        std::cout << "Folder already exists." << std::endl;
        return false;
    }

    if (fs::create_directory(folderName)) {
        std::cout << "Folder created successfully." << std::endl;
        return true;
    } else {
        std::cout << "Failed to create folder." << std::endl;
        return false;
    }
}

bool createPath(std::string const & dirName, std::error_code & err)
{
    err.clear();
    if (!std::filesystem::create_directories(dirName, err))
    {
        if (std::filesystem::exists(dirName))
        {
            // The folder already exists:
            err.clear();
            return true;    
        }
        return false;
    }
    return true;
}