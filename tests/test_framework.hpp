#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rl::tests {

using TestFunction = std::function<void()>;

struct TestCase {
    std::string name;
    TestFunction function;
};

class Registry {
  public:
    static Registry& instance() {
        static Registry registry;
        return registry;
    }

    void add(std::string name, TestFunction function) {
        tests_.push_back({std::move(name), std::move(function)});
    }

    const std::vector<TestCase>& tests() const {
        return tests_;
    }

  private:
    std::vector<TestCase> tests_;
};

class Registrar {
  public:
    Registrar(const std::string& name, TestFunction function) {
        Registry::instance().add(name, std::move(function));
    }
};

class AssertionFailure : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

inline std::string location(const char* file, int line) {
    std::ostringstream stream;
    stream << file << ':' << line;
    return stream.str();
}

template <typename T, typename = void>
struct is_streamable : std::false_type {};

template <typename T>
struct is_streamable<
    T,
    std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>>
    : std::true_type {};

template <typename T>
std::string to_debug_string(const T& value) {
    if constexpr (is_streamable<T>::value) {
        std::ostringstream stream;
        stream << value;
        return stream.str();
    } else {
        return "<non-streamable>";
    }
}

inline void fail(const std::string& message, const char* file, int line) {
    throw AssertionFailure(location(file, line) + " " + message);
}

inline void expect_true(
    const bool condition,
    const char* expression,
    const char* file,
    const int line) {
    if (!condition) {
        fail(std::string("expected true: ") + expression, file, line);
    }
}

template <typename Lhs, typename Rhs>
void expect_eq(
    const Lhs& lhs,
    const Rhs& rhs,
    const char* lhs_expression,
    const char* rhs_expression,
    const char* file,
    const int line) {
    if (!(lhs == rhs)) {
        std::ostringstream stream;
        stream << "expected equality for " << lhs_expression << " and "
               << rhs_expression << ", got [" << to_debug_string(lhs)
               << "] vs [" << to_debug_string(rhs) << "]";
        fail(stream.str(), file, line);
    }
}

inline void expect_near(
    const double lhs,
    const double rhs,
    const double tolerance,
    const char* lhs_expression,
    const char* rhs_expression,
    const char* tolerance_expression,
    const char* file,
    const int line) {
    if (std::fabs(lhs - rhs) > tolerance) {
        std::ostringstream stream;
        stream << "expected " << lhs_expression << " ~= " << rhs_expression
               << " within " << tolerance_expression << ", got [" << lhs
               << "] vs [" << rhs << "]";
        fail(stream.str(), file, line);
    }
}

inline int run_all() {
    int failures = 0;
    const auto& tests = Registry::instance().tests();

    for (const auto& test : tests) {
        try {
            test.function();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const AssertionFailure& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << " - " << error.what() << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name
                      << " - unexpected exception: " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name
                      << " - unexpected non-standard exception\n";
        }
    }

    if (failures == 0) {
        std::cout << "All tests passed (" << tests.size() << ")\n";
    } else {
        std::cerr << failures << " test(s) failed out of " << tests.size()
                  << '\n';
    }

    return failures == 0 ? 0 : 1;
}

}  // namespace rl::tests

#define RL_TEST_CONCAT_IMPL(lhs, rhs) lhs##rhs
#define RL_TEST_CONCAT(lhs, rhs) RL_TEST_CONCAT_IMPL(lhs, rhs)

#define RL_TEST_CASE(name)                                                    \
    static void RL_TEST_CONCAT(test_function_, __LINE__)();                   \
    static ::rl::tests::Registrar RL_TEST_CONCAT(test_registrar_, __LINE__)(  \
        name,                                                                 \
        &RL_TEST_CONCAT(test_function_, __LINE__));                           \
    static void RL_TEST_CONCAT(test_function_, __LINE__)()

#define RL_EXPECT_TRUE(expression)                                        \
    ::rl::tests::expect_true((expression), #expression, __FILE__, __LINE__)

#define RL_EXPECT_EQ(lhs, rhs)                                              \
    ::rl::tests::expect_eq((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__)

#define RL_EXPECT_NEAR(lhs, rhs, tolerance)                                 \
    ::rl::tests::expect_near(                                                \
        (lhs),                                                               \
        (rhs),                                                               \
        (tolerance),                                                         \
        #lhs,                                                                \
        #rhs,                                                                \
        #tolerance,                                                          \
        __FILE__,                                                            \
        __LINE__)

#define RL_EXPECT_THROW(statement)                                             \
    do {                                                                       \
        bool threw_exception = false;                                          \
        try {                                                                  \
            statement;                                                         \
        } catch (...) {                                                        \
            threw_exception = true;                                            \
        }                                                                      \
        ::rl::tests::expect_true(                                              \
            threw_exception,                                                   \
            "expected exception from " #statement,                             \
            __FILE__,                                                          \
            __LINE__);                                                         \
    } while (false)
