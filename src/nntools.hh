#ifndef NNOUT_HH
#define NNOUT_HH

// Lightweight logging helpers for nixnan
// Usage:
//   nnout() << "message" << std::endl;
//   set_out_file(filename); // optional – redirects nnout() output to file

#include <iostream>
#include <string>

// Return the current logging stream (default std::cerr) and prepend tool tag.
// Prototype required by user: std::ostream& nnout();
// NOTE: nnout() itself does not serialize writes. Building a multi-part
// message with several `<<` calls against the stream it returns is not
// safe if another thread can be doing the same concurrently (nixnan's
// exception and histogram-threshold reports each run on their own
// background thread) -- their individual `<<` calls can interleave and
// produce a torn/mangled line. Use nnout_line() for any message assembled
// from more than one `<<`, so the whole thing is written as one atomic,
// mutex-protected chunk.
std::ostream& nnout();

// Return the underlying stream WITHOUT inserting the prefix.
// Use this for querying/modifying stream flags without accidental output.
std::ostream& nnout_stream();

// Write one complete, already-assembled message as a single unit, holding a
// shared mutex for the duration -- safe to call concurrently from multiple
// threads without their output interleaving. Prepends "#nixnan: " and
// appends a trailing newline if `msg` doesn't already end in one.
void nnout_line(const std::string& msg);

// Redirect output to a file. If opening fails, stays on current stream.
// Prototype required by user: void set_out_file(std::string& filename);
void set_out_file(std::string& filename);

// Terminate the program with an optional message.
void nnterminate(const std::string& reason = "");

#endif // NNOUT_HH
