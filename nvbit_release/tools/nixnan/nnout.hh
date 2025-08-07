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
std::ostream& nnout();

// Return the underlying stream WITHOUT inserting the prefix.
// Use this for querying/modifying stream flags without accidental output.
std::ostream& nnout_stream();

// Redirect output to a file. If opening fails, stays on current stream.
// Prototype required by user: void set_out_file(std::string& filename);
void set_out_file(std::string& filename);

#endif // NNOUT_HH
