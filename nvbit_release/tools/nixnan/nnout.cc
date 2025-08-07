// Implementation for nnout() and set_out_file()
#include "nnout.hh"
#include <fstream>
#include <memory>

namespace {
struct logger_singleton {
    std::ostream* stream = &std::cerr;
    std::unique_ptr<std::ofstream> file_stream;
};

logger_singleton& get_logger() {
    static logger_singleton inst;
    return inst;
}
}

std::ostream& nnout() {
    return *(get_logger().stream) << "#nixnan: ";
}

std::ostream& nnout_stream() {
    return *(get_logger().stream);
}

void set_out_file(std::string& filename) {
    auto& lg = get_logger();
    auto fs = std::make_unique<std::ofstream>(filename);
    if (fs->is_open()) {
        lg.file_stream = std::move(fs);
        lg.stream = lg.file_stream.get();
    } else {
        nnout() << "failed to open log file '" << filename << "'" << std::endl;
    }
}