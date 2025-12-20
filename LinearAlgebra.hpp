#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H
#include <vector>
#include <cassert>
#include <memory>
#include <utility>
#include <algorithm>
#include <iostream>
namespace LinearAlgebra {
	template<typename Type>
	class Matrix {
		private:
		size_t n, m;
		std::unique_ptr<Type[]> data;
		public:
		Matrix(): n(0), m(0), data(nullptr) {}
		Matrix(size_t __n, size_t __m, const Type& x = Type(0)): n(__n), m(__m), data(std::make_unique<Type[]>(__n * __m)) { std::fill_n(data.get(), __n * __m, x); }
		Matrix(Matrix&& o) noexcept : n(o.n), m(o.m), data(std::move(o.data)) { o.n = o.m = 0; }
		Matrix& operator=(Matrix&& o) noexcept {
			if (this != &o) { n = o.n; m = o.m; data = std::move(o.data); o.n = o.m = 0; }
			return *this;
		}
		Matrix& resize(size_t __n, size_t __m, const Type& x = Type(0)) {
			n = __n, m = __m; data = std::make_unique<Type[]>(__n * __m);
			std::fill_n(data.get(), __n * __m, x);
			return *this;
		}
		Matrix(const std::vector<std::vector<Type>>& x): n(x.size()), m(x.front().size()), data(std::make_unique<Type[]>(x.size() * x.front().size())) {
			for (size_t i = 0; i < n; ++i)
				for (size_t j = 0; j < m; ++j)
					data[i * m + j] = x[i][j];
		}
		Matrix(const Matrix& o): n(o.n), m(o.m), data(nullptr) {
			if (o.data == nullptr || n == 0 || m == 0) return;
			data = std::make_unique<Type[]>(n * m);
			for (Type *ptr = o.data.get(), *end = o.data.get() + n * m, *ptrr = data.get(); ptr != end; ++ptr, ++ptrr) *ptrr = *ptr;
			// std::copy(o.data.get(), o.data.get() + n * m, data.get());
		}
		Type* operator[](size_t row) { return data.get() + row * m; }
		const Type* operator[](size_t row) const { return data.get() + row * m; }
		Matrix& operator=(const Matrix& o) { 
			if (this != &o) {
				if (o.data == nullptr || o.n == 0 || o.m == 0) {
					n = m = 0; data = nullptr;
					return *this;
				}
				n = o.n; m = o.m; data = std::make_unique<Type[]>(n * m);
				for (Type *ptr = o.data.get(), *end = o.data.get() + n * m, *ptrr = data.get(); ptr != end; ++ptr, ++ptrr) *ptrr = *ptr;
				// std::copy(o.data.get(), o.data.get() + n * m, data.get());
			}
			return *this;
		}
		Matrix& operator=(const std::vector<std::vector<Type>>& o) {
			n = o.size(); m = o.front().size(); data = std::make_unique<Type[]>(n * m);
			for (size_t i = 0; i < n; ++i)
				for (size_t j = 0; j < m; ++j)
					data[i * m + j] = o[i][j];
			return *this;
		}
		size_t N() const { return n; }
		size_t M() const { return m; }
		Type& operator () (size_t row, size_t col) { return data[row * m + col]; }
		const Type& operator () (size_t row, size_t col) const { return data[row * m + col]; }
		Matrix transpose() const {
			Matrix res(m, n);
			Type* src_row = data.get();
			for (size_t i = 0; i < n; ++i, src_row += m) {
				Type* dst = res.data.get() + i;
				Type* src = src_row;
				for (size_t j = 0; j < m; ++j, ++src, dst += n) {
					*dst = *src;
				}
			}
			return res; // RVO
		}
		Matrix& transposeSelf() {
			Matrix res(m, n);
			Type* src_row = data.get();
			for (size_t i = 0; i < n; ++i, src_row += m) {
				Type* dst = res.data.get() + i;
				Type* src = src_row;
				for (size_t j = 0; j < m; ++j, ++src, dst += n) {
					*dst = *src;
				}
			}
			return *this = std::move(res);
		}

		Matrix& operator *= (const Matrix& o) {
			Matrix res(n, o.m);
			Type *ptr = data.get();
			for (size_t i = 0; i < n; ++i) {
				Type *ptro = o.data.get();
				for (size_t k = 0; k < m; ++k, ++ptr) {
					Type tmp = *ptr;
					Type *ptrr = res.data.get() + i * o.m;
					for (size_t j = 0; j < o.m; ++j, ++ptro, ++ptrr) {
						*ptrr += tmp * *ptro;
					}
				}
			}
			return *this = std::move(res);
		}
		Matrix operator * (const Matrix& o) const {
			Matrix res(n, o.m);
			Type *ptr = data.get();
			for (size_t i = 0; i < n; ++i) {
				Type *ptro = o.data.get();
				for (size_t k = 0; k < m; ++k, ++ptr) {
					Type tmp = *ptr;
					Type *ptrr = res.data.get() + i * o.m;
					for (size_t j = 0; j < o.m; ++j, ++ptro, ++ptrr) {
						*ptrr += tmp * *ptro;
					}
				}
			}
			return res; // RVO
		}
		Matrix& operator += (const Matrix& o) {
			for (Type *ptr = data.get(), *end = data.get() + n * m, *ptro = o.data.get(); ptr != end; ++ptr, ++ptro) *ptr += *ptro;
			return *this;
		}
		Matrix operator + (const Matrix& o) const {
			Matrix res(*this);
			for (Type *ptr = res.data.get(), *end = res.data.get() + n * m, *ptro = o.data.get(); ptr != end; ++ptr, ++ptro) *ptr += *ptro;
			return res; // RVO
		}
		Matrix& operator -= (const Matrix& o) {
			for (Type *ptr = data.get(), *end = data.get() + n * m, *ptro = o.data.get(); ptr != end; ++ptr, ++ptro) *ptr -= *ptro;
			return *this;
		}
		Matrix operator - (const Matrix& o) const {
			Matrix res(*this);
			for (Type *ptr = res.data.get(), *end = res.data.get() + n * m, *ptro = o.data.get(); ptr != end; ++ptr, ++ptro) *ptr -= *ptro;
			return res; // RVO
		}
		Matrix& operator *= (const Type& o) {
			for (Type *ptr = data.get(), *end = data.get() + n * m; ptr != end; ++ptr) *ptr *= o;
			return *this;
		}
		Matrix operator * (const Type& o) const {
			Matrix res(*this);
			for (Type *ptr = res.data.get(), *end = res.data.get() + n * m; ptr != end; ++ptr) *ptr *= o;
			return res; // RVO
		}
		Matrix& operator += (const Type &o) {
			for (Type *ptr = data.get(), *end = data.get() + n * m; ptr != end; ++ptr) *ptr += o;
			return *this;
		}
		Matrix operator + (const Type &o) const {
			Matrix res(*this);
			for (Type *ptr = res.data.get(), *end = res.data.get() + n * m; ptr != end; ++ptr) *ptr += o;
			return res; // RVO
		}
		Matrix& operator -= (const Type &o) {
			for (Type *ptr = data.get(), *end = data.get() + n * m; ptr != end; ++ptr) *ptr -= o;
			return *this;
		}
		Matrix operator - (const Type &o) const {
			Matrix res(*this);
			for (Type *ptr = res.data.get(), *end = res.data.get() + n * m; ptr != end; ++ptr) *ptr -= o;
			return res; // RVO
		}
		bool operator == (const Matrix& o) const {
			if (n != o.n || m != o.m) return false;
			if (data == o.data) return true;
			if (data == nullptr || o.data == nullptr) return false;
			for (Type *ptr = data.get(), *end = data.get() + n * m, *ptro = o.data.get(); ptr != end; ++ptr, ++ptro)
				if (*ptr != *ptro) return false;
			return true;
		}
		bool operator != (const Matrix& o) const {
			if (n != o.n || m != o.m) return true;
			if (data == o.data) return false;
			if (data == nullptr || o.data == nullptr) return true;
			for (Type *ptr = data.get(), *end = data.get() + n * m, *ptro = o.data.get(); ptr != end; ++ptr, ++ptro)
				if (*ptr != *ptro) return true;
			return false;
		}
		Matrix& applyFunctionSelf(Type (*func)(Type)) { for (Type *ptr = data.get(), *end = data.get() + n * m; ptr != end; ++ptr) *ptr = func(*ptr); return *this; }
		Matrix& applyFunctionSelf(Type (*func)(const Type&)) { for (Type *ptr = data.get(), *end = data.get() + n * m; ptr != end; ++ptr) *ptr = func(*ptr); return *this; }
		Matrix applyFunction(Type (*func)(Type)) const {
			Matrix res(*this);
			for (Type *ptr = res.data.get(), *end = res.data.get() + n * m; ptr != end; ++ptr) *ptr = func(*ptr);
			return res; // RVO
		}
		Matrix applyFunction(Type (*func)(const Type&)) const {
			Matrix res(*this);
			for (Type *ptr = res.data.get(), *end = res.data.get() + n * m; ptr != end; ++ptr) *ptr = func(*ptr);
			return res; // RVO
		}
		Matrix operator%(const Matrix& o) const {
			Matrix res(n, m);
			for (Type *ptr = data.get(), *end = data.get() + n * m, *ptro = o.data.get(), *ptrr = res.data.get(); ptr != end; ++ptr, ++ptro, ++ptrr) *ptrr = *ptr * *ptro;
			return res; // RVO
		}
		Matrix& operator%=(const Matrix& o) {
			for (Type *ptr = data.get(), *end = data.get() + n * m, *ptro = o.data.get(); ptr != end; ++ptr, ++ptro) *ptr *= *ptro;
			return *this;
		}
		friend std::ostream& operator<<(std::ostream& os, const Matrix& p) {
			for (size_t i = 0; i < p.n; ++i) {
				for (size_t j = 0; j < p.m; ++j) {
					os << p(i, j) << " \n"[j == p.m - 1];
				}
				if (i != p.n - 1) os << "\n";
			}
			return os;
		}
	};
}
#endif