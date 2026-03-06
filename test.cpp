#include "MatTool.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

// ── simple test harness ──────────────────────────────────────────────────────
static int passed = 0, failed = 0;

#define CHECK(cond, name) \
    do { \
        if (cond) { std::cout << "[PASS] " << name << "\n"; ++passed; } \
        else      { std::cout << "[FAIL] " << name << "\n"; ++failed; } \
    } while(0)

#define CHECK_THROW(expr, name) \
    do { \
        bool threw = false; \
        try { (void)(expr); } catch (...) { threw = true; } \
        CHECK(threw, name); \
    } while(0)

#define CHECK_NO_THROW(expr, name) \
    do { \
        bool threw = false; \
        try { (void)(expr); } catch (...) { threw = true; } \
        CHECK(!threw, name); \
    } while(0)

static bool eq(double a, double b) { return std::fabs(a - b) < 1e-9; }

// ── tests ────────────────────────────────────────────────────────────────────

void test_default_constructor() {
    Mat m;
    CHECK(!m.isAllocated(), "default ctor: not allocated");
}

void test_size_constructor() {
    Mat m(3, 4);
    CHECK(m.Rows() == 3,     "size ctor: rows");
    CHECK(m.Cols() == 4,     "size ctor: cols");
    CHECK(m.isAllocated(),   "size ctor: allocated");
    CHECK(eq(m.get(0, 0), 0.0), "size ctor: zero-initialised");
    CHECK_THROW(Mat(0, 3),   "size ctor: throws on zero dimension");
}

void test_load_flat_array() {
    double data[] = {1, 2, 3, 4, 5, 6};
    Mat m(data, 2, 3);
    CHECK(m.Rows() == 2,        "load flat: rows");
    CHECK(m.Cols() == 3,        "load flat: cols");
    CHECK(eq(m.get(0, 0), 1.0), "load flat: [0,0]");
    CHECK(eq(m.get(1, 2), 6.0), "load flat: [1,2]");
}

void test_load_2d_array() {
    double row0[] = {1, 2, 3};
    double row1[] = {4, 5, 6};
    double* rows[] = {row0, row1};
    Mat m(rows, 2, 3);
    CHECK(eq(m.get(0, 1), 2.0), "load 2D: [0,1]");
    CHECK(eq(m.get(1, 0), 4.0), "load 2D: [1,0]");
}

void test_copy_constructor() {
    double data[] = {1, 2, 3, 4};
    Mat a(data, 2, 2);
    Mat b(a);
    CHECK(a.same(b), "copy ctor: same contents");
    // modify b, a must be unchanged
    b.set(99, 0, 0);
    CHECK(!a.same(b), "copy ctor: deep copy (independent)");
}

// Bug 1: submatrix constructor checked H/W (both 0) instead of m.H/m.W
void test_submatrix_constructor() {
    double data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Mat m(data, 3, 3);
    // used to always throw because it tested H==0 and W==0
    CHECK_NO_THROW(Mat(m, 0, 0, 1, 1), "submatrix ctor: valid range no throw");
    Mat sub(m, 0, 0, 1, 1);
    CHECK(sub.Rows() == 2 && sub.Cols() == 2, "submatrix ctor: correct size");
    CHECK(eq(sub.get(0, 0), 1.0), "submatrix ctor: [0,0]");
    CHECK(eq(sub.get(1, 1), 5.0), "submatrix ctor: [1,1]");
    CHECK_THROW(Mat(m, -1, 0, 1, 1), "submatrix ctor: negative index throws");
    CHECK_THROW(Mat(m, 0, 0, 3, 1), "submatrix ctor: out-of-bounds throws");
}

// Bug 2: function was named inverse() but performed a transpose
void test_transpose() {
    double data[] = {1, 2, 3, 4, 5, 6};
    Mat m(data, 2, 3);           // 2×3
    Mat t = m.transpose();       // should be 3×2
    CHECK(t.Rows() == 3,         "transpose: rows");
    CHECK(t.Cols() == 2,         "transpose: cols");
    CHECK(eq(t.get(0, 0), 1.0),  "transpose: [0,0]");
    CHECK(eq(t.get(1, 0), 2.0),  "transpose: [1,0]");
    CHECK(eq(t.get(2, 0), 3.0),  "transpose: [2,0]");
    CHECK(eq(t.get(0, 1), 4.0),  "transpose: [0,1]");
    // transpose of transpose == original
    Mat tt = t.transpose();
    CHECK(m.same(tt),            "transpose: T^T == original");
}

// Bug 3: load(const Mat&) had self-assignment use-after-free
void test_self_load() {
    double data[] = {1, 2, 3, 4};
    Mat m(data, 2, 2);
    // This must not crash / corrupt data
    m.load(m);
    CHECK(eq(m.get(0, 0), 1.0), "self-load: [0,0] intact");
    CHECK(eq(m.get(1, 1), 4.0), "self-load: [1,1] intact");
}

// Bug 4: sumInRow/sumInCol naming was swapped
void test_sum_in_row_col() {
    // matrix: [[1,2,3],[4,5,6]]
    double data[] = {1, 2, 3, 4, 5, 6};
    Mat m(data, 2, 3);

    // sumInRow should return one value per row: [1+2+3, 4+5+6] = [6, 15]
    double* rows = m.sumInRow();
    CHECK(m.Rows() == 2,          "sumInRow: row count");
    CHECK(eq(rows[0], 6.0),       "sumInRow: row 0 sum");
    CHECK(eq(rows[1], 15.0),      "sumInRow: row 1 sum");
    delete[] rows;

    // sumInCol should return one value per column: [1+4, 2+5, 3+6] = [5, 7, 9]
    double* cols = m.sumInCol();
    CHECK(m.Cols() == 3,          "sumInCol: col count");
    CHECK(eq(cols[0], 5.0),       "sumInCol: col 0 sum");
    CHECK(eq(cols[1], 7.0),       "sumInCol: col 1 sum");
    CHECK(eq(cols[2], 9.0),       "sumInCol: col 2 sum");
    delete[] cols;
}

void test_arithmetic() {
    double a[] = {1, 2, 3, 4};
    double b[] = {5, 6, 7, 8};
    Mat ma(a, 2, 2), mb(b, 2, 2);

    Mat r = ma.plus(mb);
    CHECK(eq(r.get(0,0), 6.0) && eq(r.get(1,1), 12.0), "plus: element-wise");

    r = mb.minus(ma);
    CHECK(eq(r.get(0,0), 4.0) && eq(r.get(1,1), 4.0),  "minus: element-wise");

    r = ma.mult(mb);
    CHECK(eq(r.get(0,0), 5.0) && eq(r.get(1,1), 32.0), "mult: element-wise");

    r = mb.div(ma);
    CHECK(eq(r.get(0,0), 5.0) && eq(r.get(1,1), 2.0),  "div: element-wise");

    Mat mc(2, 3);
    CHECK_THROW(ma.plus(mc),  "plus: dimension mismatch throws");
    CHECK_THROW(ma.minus(mc), "minus: dimension mismatch throws");
    CHECK_THROW(ma.mult(mc),  "mult: dimension mismatch throws");
    CHECK_THROW(ma.div(mc),   "div: dimension mismatch throws");
}

void test_diag_eye_ones() {
    Mat m;
    Mat id = m.eye(3);
    CHECK(id.Rows() == 3 && id.Cols() == 3, "eye: size");
    CHECK(eq(id.get(0,0), 1.0) && eq(id.get(1,1), 1.0) && eq(id.get(0,1), 0.0),
          "eye: values");

    Mat on = m.ones(2, 4);
    CHECK(on.Rows() == 2 && on.Cols() == 4, "ones: size");
    CHECK(eq(on.sum(), 8.0),                "ones: sum");

    double data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Mat sq(data, 3, 3);
    Mat d = sq.diag();
    CHECK(d.Rows() == 3 && d.Cols() == 1,   "diag: size");
    CHECK(eq(d.get(0,0), 1.0) && eq(d.get(1,0), 5.0) && eq(d.get(2,0), 9.0),
          "diag: values");
}

void test_get_row_col() {
    double data[] = {1, 2, 3, 4, 5, 6};
    Mat m(data, 2, 3);

    Mat r = m.getRow(1);
    CHECK(r.Rows() == 1 && r.Cols() == 3, "getRow: size");
    CHECK(eq(r.get(0,0), 4.0) && eq(r.get(0,2), 6.0), "getRow: values");
    CHECK_THROW(m.getRow(2), "getRow: out of range throws");

    Mat c = m.getCol(2);
    CHECK(c.Rows() == 2 && c.Cols() == 1, "getCol: size");
    CHECK(eq(c.get(0,0), 3.0) && eq(c.get(1,0), 6.0), "getCol: values");
    CHECK_THROW(m.getCol(3), "getCol: out of range throws");
}

void test_match_in_row_col() {
    double a[] = {1, 2, 3, 4};
    double b[] = {5, 6, 7, 8};
    Mat ma(a, 2, 2), mb(b, 2, 2);

    Mat h = ma.matchInRow(mb);
    CHECK(h.Rows() == 2 && h.Cols() == 4, "matchInRow: size");
    CHECK(eq(h.get(0,2), 5.0) && eq(h.get(1,3), 8.0), "matchInRow: values");

    Mat v = ma.matchInCol(mb);
    CHECK(v.Rows() == 4 && v.Cols() == 2, "matchInCol: size");
    CHECK(eq(v.get(2,0), 5.0) && eq(v.get(3,1), 8.0), "matchInCol: values");

    Mat mx(3, 2);  // 3 rows — mismatches ma (2 rows)
    CHECK_THROW(ma.matchInRow(mx), "matchInRow: row mismatch throws");
    Mat my(2, 3);  // 3 cols — mismatches ma (2 cols)
    CHECK_THROW(ma.matchInCol(my), "matchInCol: col mismatch throws");
}

void test_clone_and_same() {
    double data[] = {1, 2, 3, 4};
    Mat m(data, 2, 2);
    Mat c = m.clone();
    CHECK(m.same(c), "clone: same contents");
    c.set(99, 0, 0);
    CHECK(!m.same(c), "clone: independent copy");
}

void test_get_sub_mat() {
    double data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Mat m(data, 3, 3);
    Mat s = m.getSubMat(1, 1, 2, 2);
    CHECK(s.Rows() == 2 && s.Cols() == 2, "getSubMat: size");
    CHECK(eq(s.get(0,0), 5.0) && eq(s.get(1,1), 9.0), "getSubMat: values");
}

void test_set_and_bounds() {
    Mat m(3, 3);
    m.set(7.0, 1, 2);
    CHECK(eq(m.get(1, 2), 7.0), "set/get: value");
    CHECK_THROW(m.set(1.0, -1, 0), "set: negative row throws");
    CHECK_THROW(m.set(1.0, 3,  0), "set: row == H throws");
    CHECK_THROW(m.get(-1, 0),      "get: negative row throws");
    CHECK_THROW(m.get(0,  3),      "get: col == W throws");
}

void test_sum() {
    double data[] = {1, 2, 3, 4, 5, 6};
    Mat m(data, 2, 3);
    CHECK(eq(m.sum(), 21.0), "sum: total");
    CHECK(eq(m.sum(0, 0, 0, 2), 6.0),  "sum(rect): row 0");
    CHECK(eq(m.sum(0, 0, 1, 1), 12.0), "sum(rect): 2x2 top-left");
    CHECK_THROW(m.sum(0, 0, 2, 2), "sum(rect): out-of-bounds throws");
}

// ── main ─────────────────────────────────────────────────────────────────────

int main()
{
    test_default_constructor();
    test_size_constructor();
    test_load_flat_array();
    test_load_2d_array();
    test_copy_constructor();
    test_submatrix_constructor();   // Bug 1
    test_transpose();               // Bug 2
    test_self_load();               // Bug 3
    test_sum_in_row_col();          // Bug 4
    test_arithmetic();
    test_diag_eye_ones();
    test_get_row_col();
    test_match_in_row_col();
    test_clone_and_same();
    test_get_sub_mat();
    test_set_and_bounds();
    test_sum();

    std::cout << "\nResults: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
