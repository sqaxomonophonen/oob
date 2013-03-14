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

// tree size in "pixels"
#define TX (10)
#define TS (1<<TX)
#define TN (1<<(TX+TX))

// leaf size in "pixels"
#define LX (3)
#define LS (1<<LX)
#define LN (1<<(LX+LX))

// tree size in leafs
#define DX (TX-LX)
#define DS (1<<(DX))
#define DN (1<<(DX+DX))

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
	vec2(float x, float y) : x(x), y(y) {}

	struct vec2 operator-(const struct vec2& other) const {
		return vec2(x - other.x, y - other.y);
	}
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

struct mass {
	float m0;
	float c0;
	float c1[2];

	void clear() {
		m0 = 0;
		c0 = 0;
		c1[0] = 0;
		c1[1] = 0;
	}
};

struct leaf {
	struct mass masses[LN];

	void clear() {
		for(int i = 0; i < LN; i++) masses[i].clear();
	}
};

struct cell {
	int r[4]; // children
	float m0; // mass
	struct vec2 z; // center of mass
	float m2[3]; // specific quadrupole moment (m2/m0)
	int n;
	float rmax;
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
		n = 0;
		ti = -1; // XXX or 0? to have bzero support?
	}

	/* updates total mass, updates partial center of mass */
	void pass1(float x, float y, float m) {
		n++;
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

	void pass01(float zx, float zy, float m) {
		m0 += m;
		z.x += zx;
		z.y += zy;
	}

	void pass02(float x0, float y0, struct leaf& leaf) {
		int p = 0;
		for(float dy = 0; dy < LS; dy+=1) {
			for(float dx = 0; dx < LS; dx+=1) {
				float m = leaf.masses[p++].m0;
				float x = x0 + dx;
				float y = y0 + dy;
				float qx = x - z.x;
				float qy = y - z.y;
				float im0 = m / m0;
				m2[0] += im0 * qx * qx;
				m2[1] += im0 * qy * qy;
				m2[2] += im0 * qx * qy;
			}
		}
	}
};

struct tree {
	struct lst<struct cell> cells;
	struct lst<struct leaf> leafs;
	struct lst<struct taylor_coefficients> tcs;

	struct cell empty_cell;
	struct leaf empty_leaf;

	tree() {
		empty_cell.clear();
		empty_leaf.clear();
	}

	void reset() {
		cells.reset();
		cells.write(&empty_cell);
		leafs.reset();
		leafs.write(&empty_leaf); // hack.. to prevent zero index in cell.r
		tcs.reset();
	}

	void pass1(int x, int y, float m) {
		unsigned int cursor = 0;
		float fx = x;
		float fy = y;
		int mask = (1<<(TX-1))-1;
		for(int i = (TX-1); i >= (LX-1); i--) {
			struct cell& c = *cells[cursor];
			c.pass1(fx, fy, m);
			int q = ((x>>i)?1:0)+((y>>i)?2:0);
			int r = c.r[q];
			if(i > (LX-1)) {
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
			} else {
				if(r == 0) {
					leafs.write(&empty_leaf);
					r = cells[cursor]->r[q] = leafs.last();
				}
				struct leaf& leaf = *leafs[r];
				unsigned int mi = x + y * LS;
				leaf.masses[mi].m0 += m;
				return;
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
		int mask = (1<<(TX-1))-1;
		for(int i = (TX-1); i >= (LX-1); i--) {
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
			for(int y = 0; y < TS; y++) {
				for(int x = 0; x < TS; x++) {
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
			for(int y = 0; y < TS; y++) {
				for(int x = 0; x < TS; x++) {
					float m = init[ptr++];
					if(m > 0.0f) {
						pass2(x, y, m);
					}
				}
			}
		}
	}

	void pass01(int x, int y, float zm, float zx, float zy, struct leaf& leaf) {
		unsigned int cursor = 0;
		int mask = (1<<(DX-1))-1;
		for(int i = (DX-1); i >= 0; i--) {
			//printf("i:%d x:%d y:%d mask:%d\n",i,x,y,mask);
			struct cell& c = *cells[cursor];
			c.pass01(zx, zy, zm);
			int q = ((x>>i)?1:0)+((y>>i)?2:0);
			int r = c.r[q];
			if(i >= 0) {
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
			} else {
				leafs.write(&leaf);
				cells[cursor]->r[q] = leafs.last();
				//printf("\n");
				return;
			}
			x &= mask;
			y &= mask;
			mask >>= 1;
		}
	}

	void zz_rec(int i, float x0, float y0, float s) {
		struct cell& c = *cells[i];
		if(c.m0 == 0) return;
		//printf("%f %f %f\n", c.m0, c.z.x, c.z.y);
		c.normalize_z();
		float h = s*0.5f;
		float x0h = x0 + h;
		float y0h = y0 + h;
		float dx;
		float dy;
		if(c.z.x <= (x0h)) {
			if(c.z.y <= (y0h)) {
				dx = c.z.x - (x0+s);
				dy = c.z.y - (y0+s);
			} else {
				dx = c.z.x - (x0+s);
				dy = c.z.y - y0;
			}
		} else {
			if(c.z.y <= (y0h)) {
				dx = c.z.x - x0;
				dy = c.z.y - (y0+s);
			} else {
				dx = c.z.x - x0;
				dy = c.z.y - y0;
			}
		}
		c.rmax = sqrtf(dx*dx+dy*dy);
		if(s >= LS) {
			if(c.r[0]) zz_rec(c.r[0], x0, y0, h);
			if(c.r[1]) zz_rec(c.r[1], x0h, y0, h);
			if(c.r[2]) zz_rec(c.r[2], x0, y0h, h);
			if(c.r[3]) zz_rec(c.r[3], x0h, y0h, h);
		}
	}

	void zz() {
		zz_rec(0, 0, 0, TS);
	}

	void pass02(int x, int y, struct leaf& leaf) {
		unsigned int cursor = 0;
		float x0 = x*LS;
		float y0 = y*LS;
		int mask = (1<<(DX-1))-1;
		for(int i = (DX-1); i >= 0; i--) {
			struct cell& c = *cells[cursor];
			c.pass02(x0, y0, leaf);
			int q = ((x>>i)?1:0)+((y>>i)?2:0);
			cursor = c.r[q];
			x &= mask;
			y &= mask;
			mask >>= 1;
		}
	}

	void build2(float* init) {
		reset();
		{
			scope_timer t0("pass01");
			int p = 0;
			for(int y = 0; y < DS; y++) {
				for(int x = 0; x < DS; x++) {
					struct leaf leaf;
					leaf.clear();
					int lp = 0;
					int pp = p;
					float x0 = x * LS;
					float y0 = y * LS;
					float zm = 0;
					float zx = 0;
					float zy = 0;
					for(float ly = 0; ly < LS; ly+=1) {
						for(float lx = 0; lx < LS; lx+=1) {
							float m = init[pp++];
							leaf.masses[lp++].m0 = m;
							zm += m;
							zx += m * (x0 + lx);
							zy += m * (y0 + ly);
						}
						pp += TS-LS;
					}
					if(zm > 0) {
						pass01(x, y, zm, zx, zy, leaf);
					}
					p += LS;
				}
				p += (LS-1) * TS;
			}
		}
		printf("number of cells: %d\n", cells.last());
		{
			scope_timer t0("zz");
			zz();
		}
		{
			scope_timer t0("pass02");
			int p = 0;
			for(int y = 0; y < DS; y++) {
				for(int x = 0; x < DS; x++) {
					struct leaf leaf;
					leaf.clear();
					int lp = 0;
					int pp = p;
					float zm = 0;
					for(float ly = 0; ly < LS; ly+=1) {
						for(float lx = 0; lx < LS; lx+=1) {
							float m = init[pp++];
							leaf.masses[lp++].m0 = m;
							zm += m;
						}
						pp += TS-LS;
					}
					if(zm > 0) {
						pass02(x, y, leaf);
					}
					p += LS;
				}
				p += (LS-1) * TS;
			}
		}
	}

	void interact_rec(int a_index, int a_x0, int a_y0, int a_level, int b_index, int b_x0, int b_y0, int b_level) {
		struct cell& a = *cells[a_index];
		struct cell& b = *cells[b_index];
		if(a.m0 == 0) return;
		if(b.m0 == 0) return;
		//struct vec2 r = b.z - a.z;
		//float rsqr = r.lensqr();


		//unsigned long n1n2 = ((unsigned long) a.n) * ((unsigned long) b.n);
		//printf("%lu %d %d\n", n1n2, a.n, b.n);
	}

	void interact() {
		scope_timer t0("interact");
		interact_rec(0,0,0,TX,0,0,0,TX);
	}

	void evaluate() {
		scope_timer t0("evaluate");
	}
};

int main(int argc, char** argv) {
	if(argc != 2) {
		fprintf(stderr, "usage: %s <1024x1024.gray>\n", argv[0]);
		exit(1);
	}

	char* initc = (char*) malloc(TN);
	int fd = open(argv[1], O_RDONLY);
	readn(fd, initc, TN);
	close(fd);

	float* init = (float*) malloc(TN * sizeof(float));
	for(int i = 0; i < TN; i++) init[i] = (float) initc[i];

	struct tree t;
	{
		scope_timer t0("warm up");
		t.build2(init);
		t.interact();
	}
	printf("\n\n-------------------\n");
	{
		scope_timer t0("build");
		t.build2(init);
		t.interact();
		t.evaluate();
	}

	return 0;
}

