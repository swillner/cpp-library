#ifndef NVECTOR_H
#define NVECTOR_H

#include <vector>

template<typename T, unsigned char dim, class Storage = std::vector<T>>
class nvector {
  protected:
    std::array<size_t, dim> dims;
    Storage data;

    template<unsigned char c, typename... Args>
    inline T& i_(const std::size_t index, const std::size_t& i, Args... args) noexcept {
        return i_<c + 1>(index * dims[c] + i, args...);
    }

    template<unsigned char c>
    inline T& i_(const std::size_t index) noexcept {
        static_assert(c == dim, "wrong number of arguments");
        return data[index];
    }

    template<unsigned char c, typename... Args>
    inline T& at_(const std::size_t index, const std::size_t& i, Args... args) {
        if (i >= dims[c]) {
            throw std::out_of_range("index out of bounds");
        }
        return at_<c + 1>(index * dims[c] + i, args...);
    }

    template<unsigned char c>
    inline T& at_(const std::size_t index) {
        static_assert(c == dim, "wrong number of arguments");
        return data.at(index);
    }

    template<unsigned char c, typename... Args>
    inline void initialize_(const T& initial_value, const std::size_t size, const std::size_t& i, Args... args) {
        dims[c] = i;
        initialize_<c + 1>(initial_value, size * i, args...);
    }

    template<unsigned char c>
    inline void initialize_(const T& initial_value, const std::size_t size) {
        static_assert(c == dim, "wrong number of arguments");
        data.resize(size, initial_value);
    }

  public:
    template<typename... Args>
    nvector(T initial_value, Args... args) {
        initialize_<0>(initial_value, 1, args...);
    }

    template<typename... Args>
    inline T& operator()(Args... args) noexcept {
        return i_<0>(0, args...);
    }

    template<typename... Args>
    inline T& at(Args... args) {
        return at_<0>(0, args...);
    }

    inline T* raw() { return &data[0]; }
};

#endif
