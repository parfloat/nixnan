// Implementation for nnout() and set_out_file() using simple file-scope statics
#include "nnout.hh"
#include <fstream>
#include <memory>

static std::ostream* g_nnout_stream = &std::cerr;
static std::unique_ptr<std::ofstream> g_nnout_file;

std::ostream& nnout() {
    return *g_nnout_stream << "#nixnan: ";
}

std::ostream& nnout_stream() {
    return *g_nnout_stream;
}

void set_out_file(std::string& filename) {
    auto fs = std::make_unique<std::ofstream>(filename);
    if (fs->is_open()) {
        g_nnout_file = std::move(fs);
        g_nnout_stream = g_nnout_file.get();
    } else {
        nnout() << "failed to open log file '" << filename << "'" << std::endl;
    }
}