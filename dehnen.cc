#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include "nanotime.h"

#define X (10)
#define S (1<<X)
#define N (S*S)

ssize_t readn(int fd, void *vptr, size_t n) {
	size_t  nleft;
	ssize_t nread;
	char   *ptr;

	ptr = (char*) vptr;
	nleft = n;
	while (nleft > 0) {
		if ( (nread = read(fd, ptr, nleft)) < 0) {
			if (errno == EINTR) {
				nread = 0;      /* and call read() again */
			} else {
				return (-1);
			}
		} else if (nread == 0) {
			break;              /* EOF */
		}

		nleft -= nread;
		ptr += nread;
	}
	return (n - nleft);         /* return >= 0 */
}

struct scope_timer {
	const char* timee;
	uint64_t t0;
	scope_timer(const char* timee) : timee(timee) {
		t0 = nanotime_get();
	}
	~scope_timer() {
		uint64_t dt = nanotime_get() - t0;
		printf("%s: %f ms\n", timee, dt/1e6f);
	}
};



/* growing list */
template <class T>
struct lst {
	T* elements;
	unsigned int cap;
	unsigned int size;

	lst() {
		elements = NULL;
		cap = 0;
		reset();
	}

	~lst() {
		if(elements) free(elements);
	}

	void reset() {
		size = 0;
	}

	int cap_calc(size_t cap, size_t required) {
		/* return max(cap*2, 2048, required) */
		return (cap<<1) > required ? (cap<<1) : (required > 2048 ? required : 2048);
	}

	void ensure(size_t required) {
		if(required > cap) {
			cap = cap_calc(cap, required);
			elements = (T*) realloc(elements, cap * sizeof(T));
		}
		size = required;
	}

	void write(T* e) {
		ensure(size + 1);
		memcpy(top(), e, sizeof(T));
	}

	T* operator[](int idx) {
		return elements + idx;
	}
	int last() const {
		return size - 1;
	}
	T* top() {
		return elements + size - 1;
	}
	T* operator->() {
		return elements;
	}
};


struct vec2 {
	float x,y;

	vec2() : x(0), y(0) {}
};

struct taylor_coefficients {
	float c0;
	float c1[2];
	float c2[3];
	float c3[4];

	taylor_coefficients() {
		c0 = 0;

		c1[0] = 0;
		c1[1] = 0;

		c2[0] = 0;
		c2[1] = 0;
		c2[2] = 0;

		c3[0] = 0;
		c3[1] = 0;
		c3[2] = 0;
		c3[3] = 0;
	}

	struct taylor_coefficients translate(struct vec2& dz) const {
		// TODO
		struct taylor_coefficients t;
		return t;
	}
};

struct cell {
	int r[4]; // children
	float m0; // mass
	struct vec2 z; // center of mass
	float m2[3]; // specific quadrupole moment (m2/m0)
	int ti; // taylor coefficients index

	void clear() {
		r[0] = 0;
		r[1] = 0;
		r[2] = 0;
		r[3] = 0;
		m0 = 0;
		z.x = 0;
		z.y = 0;
		m2[0] = 0;
		m2[1] = 0;
		m2[2] = 0;
		ti = -1; // XXX or 0? to have bzero support?
	}

	/* updates total mass, updates partial center of mass */
	void pass1(float x, float y, float m) {
		m0 += m;
		z.x += x * m;
		z.y += y * m;
	}

	/* calculates real center of mass */
	void normalize_z() {
		z.x /= m0;
		z.y /= m0;
	}

	/* updates specific quadrupole moment */
	void pass2(float x, float y, float m) {
		float dx = x - z.x;
		float dy = y - z.y;
		float im0 = m / m0;
		m2[0] += im0 * dx * dx;
		m2[1] += im0 * dy * dy;
		m2[2] += im0 * dx * dy;
	}
};

struct tree {
	struct lst<struct cell> cells;
	struct lst<struct taylor_coefficients> tcs;

	struct cell empty_cell;

	tree() {
		empty_cell.clear();
	}

	void reset() {
		cells.reset();
		cells.write(&empty_cell);
		tcs.reset();
	}

	void pass1(int x, int y, float m) {
		unsigned int cursor = 0;
		float fx = x;
		float fy = y;
		int mask = (1<<(X-1))-1;
		for(int i = (X-1); i >= 0; i--) {
			struct cell& c = *cells[cursor];
			c.pass1(fx, fy, m);
			int q = ((x>>i)?1:0)+((y>>i)?2:0);
			int r = c.r[q];
			if(r == 0) {
				// empty; make new cell
				cells.write(&empty_cell);
				int new_cursor = cells.last();
				cells[cursor]->r[q] = new_cursor;
				cursor = new_cursor;
			} else {
				// already a cell; enter
				cursor = r;
			}
			x &= mask;
			y &= mask;
			mask >>= 1;
		}
	}

	void pass2(int x, int y, float m) {
		unsigned int cursor = 0;
		float fx = x;
		float fy = y;
		int mask = (1<<(X-1))-1;
		for(int i = (X-1); i >= 0; i--) {
			struct cell& c = *cells[cursor];
			c.pass2(fx, fy, m);
			int q = ((x>>i)?1:0)+((y>>i)?2:0);
			cursor = c.r[q];
			x &= mask;
			y &= mask;
			mask >>= 1;
		}
	}

	void normalize_z() {
		for(int i = 0; i < cells.size; i++) {
			struct cell& c = *cells[i];
			c.normalize_z();
		}
	}

	void build(float* init) {
		reset();
		int nmass = 0;
		{
			scope_timer t0("pass1");
			int ptr = 0;
			for(int y = 0; y < S; y++) {
				for(int x = 0; x < S; x++) {
					float m = init[ptr++];
					if(m > 0.0f) {
						pass1(x, y, m);
						nmass++;
					}
				}
			}
		}
		printf("number of cells: %d / number of non-zero masses: %d\n", cells.last(), nmass);
		{
			scope_timer t0("normalize z");
			normalize_z();
		}
		{
			scope_timer t0("pass2");
			int ptr = 0;
			for(int y = 0; y < S; y++) {
				for(int x = 0; x < S; x++) {
					float m = init[ptr++];
					if(m > 0.0f) {
						pass2(x, y, m);
					}
				}
			}
		}
	}

	void interact(int a, int b) {
		// XXX try recursive, or?
	}

	void interact() {
		scope_timer t0("interact");
		interact(0,0);
	}

	void evaluate() {
		scope_timer t0("evaluate");
	}
};

/*
taylor coefficients
*/

int main(int argc, char** argv) {
	if(argc != 2) {
		fprintf(stderr, "usage: %s <1024x1024.gray>\n", argv[0]);
		exit(1);
	}

	char* initc = (char*) malloc(N);
	int fd = open(argv[1], O_RDONLY);
	readn(fd, initc, N);
	close(fd);

	float* init = (float*) malloc(N * sizeof(float));
	for(int i = 0; i < N; i++) init[i] = (float) initc[i];

	struct tree t;
	{
		scope_timer t0("warm up");
		t.build(init);
		t.interact();
	}
	printf("\n\n-------------------\n");
	{
		scope_timer t0("build");
		t.build(init);
		t.interact();
		t.evaluate();
	}

	return 0;
}

