#include "common.cuh"

std::unordered_set<std::string> read_from_file(std::string filename) {
    try {
      std::ifstream f(filename.c_str());
      std::unordered_set<std::string> res;
      if (!f) {
        std::cerr << "#nixnan: Could not open " << filename << "!" << std::endl;
      } else {
        std::string line;
        while (std::getline(f, line)) {
          res.insert(line);
        }
      }
      return res;
    } catch (const std::exception &ex) {
      std::cerr << "Exception: '" << ex.what() << "'!" << std::endl;
      exit(1);
    }
  } 