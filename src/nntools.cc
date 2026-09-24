// Implementation for nnout() and set_out_file() using simple file-scope statics
#include "nntools.hh"
#include <fstream>
#include <memory>
#include <mutex>
#include <csignal>

static std::ostream* g_nnout_stream = &std::cerr;
static std::unique_ptr<std::ofstream> g_nnout_file;
static std::mutex g_nnout_mutex;

std::ostream& nnout() {
    return *g_nnout_stream << "#nixnan: ";
}

std::ostream& nnout_stream() {
    return *g_nnout_stream;
}

void nnout_line(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_nnout_mutex);
    *g_nnout_stream << "#nixnan: " << msg;
    if (msg.empty() || msg.back() != '\n') {
        *g_nnout_stream << "\n";
    }
    g_nnout_stream->flush();
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

void nnterminate(const std::string& reason) {
    nnout_line("Terminating early: " + reason);
    std::raise(SIGTERM);
}