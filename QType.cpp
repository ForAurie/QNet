// #include <bits/stdc++.h>

// using namespace std;
// float sigmoid(float x) {
//     return 1 / (1 + exp(-x));
// }
// float f(float a, float x) {
//     return  a / (a + (1 - a) * exp(-x));
// }
// int main() {
//     for (float i = -3; i <= 3; i += 0.01) {
//         for (float j = -3; j <= 3; j += 0.01) {
//             cout << sigmoid(i + j) << ' ' << f(sigmoid(i), j) << endl;
//         }
//     }
//     return 0;
// }
// #include <cstdint>
#include <bits/stdc++.h>
constexpr std::uint16_t __MASK = 0b1111111111;
constexpr double __V = 15;
double __sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

    constexpr std::uint16_t init() {
        std::uint16_t arr[__MASK + 1]{};
        for (std::size_t i = 0; i <= __MASK; ++i) {
            arr[i] = i;
        }
    }

// constexpr auto __PRESIGMOID[] = {
//     std::uint16_t arr[__MASK + 1]{};
//     for (std::size_t i = 0; i <= __MASK; ++i) {
//         arr[i] = i;
//     }
//     return arr;
// }();
constexpr auto __PRESIGMOID = [] {
    std::uint16_t arr[__MASK + 1]{};
    for (std::size_t i = 0; i <= __MASK; ++i) {
        arr[i] = i;
    }
    return arr;
}();

class QType {
    private:
        std::uint16_t v;        
        inline bool sign() const {
            return v >> 10;
        }
        inline std::uint16_t val() const {
            return v & __MASK;
        }
    public:
        float tofloat() const {
            if (sign()) return -(__V * val() / __MASK);
            else return __V * val() / __MASK;
        }
        QType(): v(0) {}
        QType(std::uint16_t v) : v(v) {}
        QType(float x) : v(x >= 0 ? 0 : 0b10000000000) {
            if (x < 0) x = -x;
            v |= std::min(__MASK, (std::uint16_t) std::round(x / __V * __MASK));
        }
};