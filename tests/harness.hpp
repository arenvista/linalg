// Shared test harness: runs each test function, turning the library's
// "not implemented: ..." stub throws into ctest SKIPs instead of failures.
//
// Usage:
//   run("name", test_fn);   // any LinalgError starting with "not implemented"
//                           // marks the test skipped; anything else fails
//   return finish("suite"); // exits kSkipExit if anything was skipped, so
//                           // ctest (via SKIP_RETURN_CODE 77) reports it
//
// Assertion failures inside a test still abort() as in the other suites.
// As methods get implemented, tests flip from SKIP to passing on their own.
#pragma once

#include "linalg/core/Exceptions.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace testharness {

inline int passed = 0;
inline int skipped = 0;

inline constexpr int kSkipExit = 77;

template <typename Fn> void run(const char *name, Fn fn) {
    try {
        fn();
        ++passed;
    } catch (const linalg::LinalgError &e) {
        const std::string what = e.what();
        if (what.rfind("not implemented", 0) == 0) {
            std::printf("SKIP %s (%s)\n", name, what.c_str());
            ++skipped;
            return;
        }
        std::printf("FAIL %s: unexpected LinalgError: %s\n", name,
                    what.c_str());
        std::exit(1);
    } catch (const std::exception &e) {
        std::printf("FAIL %s: unexpected exception: %s\n", name, e.what());
        std::exit(1);
    }
}

inline int finish(const char *suite) {
    std::printf("%s: %d passed, %d skipped\n", suite, passed, skipped);
    return skipped > 0 ? kSkipExit : 0;
}

} // namespace testharness
