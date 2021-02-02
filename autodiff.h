/*
  Copyright (C) 2017 Sven Willner <sven.willner@gmail.com>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as published
  by the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <math.h>

#include <valarray>
#include <vector>

namespace autodiff {

template<typename T, typename Vector = std::valarray<T>>
class Variable;
template<typename T, typename Vector = std::valarray<T>>
class Value;

}  // namespace autodiff

namespace std {

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> pow(const autodiff::Value<T, Vector>& lhs, const autodiff::Value<T, Vector>& rhs);
template<typename T, typename Vector>
inline autodiff::Value<T, Vector> pow(const T& val, const autodiff::Value<T, Vector>& rhs);
template<typename T, typename Vector>
inline autodiff::Value<T, Vector> pow(const autodiff::Value<T, Vector>& lhs, const T& val);

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> log(const autodiff::Value<T, Vector>& v);
template<typename T, typename Vector>
inline autodiff::Value<T, Vector> log2(const autodiff::Value<T, Vector>& v);
template<typename T, typename Vector>
inline autodiff::Value<T, Vector> log10(const autodiff::Value<T, Vector>& v);
template<typename T, typename Vector>
inline autodiff::Value<T, Vector> exp(const autodiff::Value<T, Vector>& v);

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> min(const T& val, const autodiff::Value<T, Vector>& rhs);
template<typename T, typename Vector>
inline autodiff::Value<T, Vector> min(const autodiff::Value<T, Vector>& lhs, const T& val);

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> max(const T& val, const autodiff::Value<T, Vector>& rhs);
template<typename T, typename Vector>
inline autodiff::Value<T, Vector> max(const autodiff::Value<T, Vector>& lhs, const T& val);

}  // namespace std

namespace autodiff {

template<typename T, typename Vector>
class Value final {
    friend class Variable<T, Vector>;

  private:
    T val;
    Vector dev;
    Value(T v, Vector dev_p) : val(std::move(v)), dev(std::move(dev_p)) {}

  public:
    Value(std::size_t n, T v) : val(std::move(v)), dev(0.0, n) {}
    Value(std::size_t i, std::size_t n, T v) : val(std::move(v)), dev(0.0, n) { dev[i] = 1; }

    std::size_t size() const { return dev.size(); }
    const T& value() const { return val; }
    const Vector& derivative() const { return dev; }
    explicit operator T() const { return val; }

    Value operator-() const { return {-val, -dev}; }

    friend Value operator+(const Value& lhs, const Value& rhs) { return {lhs.val + rhs.val, lhs.dev + rhs.dev}; }
    friend Value operator+(const T& v, const Value& rhs) { return {v + rhs.val, rhs.dev}; }
    friend Value operator+(const Value& lhs, const T& v) { return {lhs.val + v, lhs.dev}; }

    friend Value operator-(const Value& lhs, const Value& rhs) { return {lhs.val - rhs.val, lhs.dev - rhs.dev}; }
    friend Value operator-(const T& v, const Value& rhs) { return {v - rhs.val, -rhs.dev}; }
    friend Value operator-(const Value& lhs, const T& v) { return {lhs.val - v, lhs.dev}; }

    friend Value operator*(const Value& lhs, const Value& rhs) { return {lhs.val * rhs.val, lhs.dev * rhs.val + lhs.val * rhs.dev}; }
    friend Value operator*(const T& v, const Value& rhs) { return {v * rhs.val, v * rhs.dev}; }
    friend Value operator*(const Value& lhs, const T& val) { return {lhs.val * val, lhs.dev * val}; }

    friend Value operator/(const Value& lhs, const Value& rhs) { return {lhs.val / rhs.val, lhs.dev / rhs.val - rhs.dev * (lhs.val / rhs.val / rhs.val)}; }
    friend Value operator/(const T& v, const Value& rhs) { return {v / rhs.val, rhs.dev * (-v / rhs.val / rhs.val)}; }
    friend Value operator/(const Value& lhs, const T& v) { return {lhs.val / v, lhs.dev / v}; }

    Value& operator+=(const Value& v) {
        val += v.val;
        dev += v.dev;
        return *this;
    }
    Value& operator+=(const T& v) {
        val += v;
        return *this;
    }

    Value& operator-=(const Value& v) {
        val -= v.val;
        dev -= v.dev;
        return *this;
    }
    Value& operator-=(const T& v) {
        val -= v;
        return *this;
    }

    Value& operator*=(const Value& v) {
        val *= v.val;
        dev = dev * v.val + v.dev * val;
        return *this;
    }
    Value& operator*=(const T& v) {
        val *= v;
        dev *= v;
        return *this;
    }

    Value& operator/=(const Value& v) {
        val /= v.val;
        dev = dev / v.val - v.dev * val / v.val / v.val;
        return *this;
    }
    Value& operator/=(const T& v) {
        val /= v;
        dev /= v;
        return *this;
    }

    friend bool operator<(const Value& lhs, const Value& rhs) { return lhs.val < rhs.val; }
    friend bool operator<(const T& v, const Value& rhs) { return v < rhs.val; }
    friend bool operator<(const Value& lhs, const T& v) { return lhs.val < v; }

    friend bool operator<=(const Value& lhs, const Value& rhs) { return lhs.val <= rhs.val; }
    friend bool operator<=(const T& v, const Value& rhs) { return v <= rhs.val; }
    friend bool operator<=(const Value& lhs, const T& v) { return lhs.val <= v; }

    friend bool operator>(const Value& lhs, const Value& rhs) { return lhs.val > rhs.val; }
    friend bool operator>(const T& v, const Value& rhs) { return v > rhs.val; }
    friend bool operator>(const Value& lhs, const T& v) { return lhs.val > v; }

    friend bool operator>=(const Value& lhs, const Value& rhs) { return lhs.val >= rhs.val; }
    friend bool operator>=(const T& v, const Value& rhs) { return v >= rhs.val; }
    friend bool operator>=(const Value& lhs, const T& v) { return lhs.val >= v; }

    friend bool operator==(const Value& lhs, const Value& rhs) { return lhs.val == rhs.val; }
    friend bool operator==(const T& v, const Value& rhs) { return v == rhs.val; }
    friend bool operator==(const Value& lhs, const T& v) { return lhs.val == v; }

    friend bool operator!=(const Value& lhs, const Value& rhs) { return lhs.val != rhs.val; }
    friend bool operator!=(const T& v, const Value& rhs) { return v != rhs.val; }
    friend bool operator!=(const Value& lhs, const T& v) { return lhs.val != v; }

    friend Value std::pow<T, Vector>(const Value& lhs, const Value& rhs);
    friend Value std::pow<T, Vector>(const T& v, const Value& rhs);
    friend Value std::pow<T, Vector>(const Value& lhs, const T& v);

    friend Value std::log<T, Vector>(const Value& v);
    friend Value std::log2<T, Vector>(const Value& v);
    friend Value std::log10<T, Vector>(const Value& v);
    friend Value std::exp<T, Vector>(const Value& v);

    friend Value std::min<T, Vector>(const T& v, const Value& rhs);
    friend Value std::min<T, Vector>(const Value& lhs, const T& v);

    friend Value std::max<T, Vector>(const T& v, const Value& rhs);
    friend Value std::max<T, Vector>(const Value& lhs, const T& v);
};

template<typename T, typename Vector>
class Variable final {
  private:
    std::vector<T> val;
    std::size_t variables_num;
    std::size_t variables_offset;

  public:
    Variable(std::size_t offset, std::size_t num, std::size_t length, T initial_value)
        : variables_num(num), variables_offset(offset), val(length, std::move(initial_value)) {}
    Variable(std::size_t num, T initial_value) : Variable(0, num, num, std::move(initial_value)) {}

    Variable& operator=(const std::vector<T>& v) {
        val = v;
        return *this;
    }
    Variable& operator=(std::vector<T>&& v) {
        val = std::forward(v);
        return *this;
    }

    std::size_t size() const { return val.size(); }

    std::vector<T>& value() { return val; }
    const std::vector<T>& value() const { return val; }

    Value<T, Vector> operator[](std::size_t i) const {
        if (variables_offset < variables_num) {
            return {i + variables_offset, variables_num, val[i]};
        } else {
            return {variables_num, val[i]};
        }
    }
    Value<T, Vector> at(std::size_t i) const {
        if (variables_offset < variables_num) {
            return {i + variables_offset, variables_num, val.at(i)};
        } else {
            return {variables_num, val.at(i)};
        }
    }
};

}  // namespace autodiff

namespace std {

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> pow(const autodiff::Value<T, Vector>& lhs, const autodiff::Value<T, Vector>& rhs) {
    const T p = pow(lhs.val, rhs.val);
    return {p, rhs.dev * (log(lhs.val) * p) + lhs.dev * (p * rhs.val / lhs.val)};
}

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> pow(const T& v, const autodiff::Value<T, Vector>& rhs) {
    const T p = pow(v, rhs.val);
    return {p, rhs.dev * (p * log(v))};
}

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> pow(const autodiff::Value<T, Vector>& lhs, const T& v) {
    return {pow(lhs.val, v), lhs.dev * (v * pow(lhs.val, v - 1))};
}

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> log(const autodiff::Value<T, Vector>& v) {
    return {log(v.val), v.dev / v.val};
}

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> log2(const autodiff::Value<T, Vector>& v) {
    return {log2(v.val), v.dev / (v.val * M_LN2)};
}

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> log10(const autodiff::Value<T, Vector>& v) {
    return {log10(v.val), v.dev / (v.val * M_LN10)};
}

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> exp(const autodiff::Value<T, Vector>& v) {
    const T e = exp(v.val);
    return {e, v.dev * e};
}

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> min(const T& v, const autodiff::Value<T, Vector>& rhs) {
    if (v < rhs.val) {
        return {rhs.dev.size(), v};
    } else {
        return rhs;
    }
}

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> min(const autodiff::Value<T, Vector>& lhs, const T& v) {
    if (lhs.val < v) {
        return lhs;
    } else {
        return {lhs.dev.size(), v};
    }
}

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> max(const T& v, const autodiff::Value<T, Vector>& rhs) {
    if (v < rhs.val) {
        return rhs;
    } else {
        return {rhs.dev.size(), v};
    }
}

template<typename T, typename Vector>
inline autodiff::Value<T, Vector> max(const autodiff::Value<T, Vector>& lhs, const T& v) {
    if (lhs.val < v) {
        return {lhs.dev.size(), v};
    } else {
        return lhs;
    }
}

}  // namespace std

#endif
