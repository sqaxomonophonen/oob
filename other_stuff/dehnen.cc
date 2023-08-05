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

template <typename N>
N max(N a, N b) { return a > b ? a : b; }

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

	float square_distance(struct vec2& other) const {
		float dx = x - other.x;
		float dy = y - other.y;
		return dx*dx + dy*dy;
	}

	float distance(struct vec2& other) const {
		return sqrtf(square_distance(other));
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

	/* calculates real center of mass */
	void normalize_z() {
		z.x /= m0;
		z.y /= m0;
	}

	void pass01(float zx, float zy, float m, int nn) {
		m0 += m;
		z.x += zx;
		z.y += zy;
		n += nn;
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

	void pass01(int x, int y, float zm, float zx, float zy, struct leaf& leaf, int n) {
		unsigned int cursor = 0;
		int mask = (1<<(DX-1))-1;
		for(int i = (DX-1); i >= 0; i--) {
			//printf("i:%d x:%d y:%d mask:%d\n",i,x,y,mask);
			struct cell& c = *cells[cursor];
			c.pass01(zx, zy, zm, n);
			int q = ((x>>i)?1:0)+((y>>i)?2:0);
			if(i > 0) {
				if(c.r[q] == 0) {
					// empty; make new cell
					cells.write(&empty_cell);
					int new_cursor = cells.last();
					cells[cursor]->r[q] = new_cursor;
					cursor = new_cursor;
				} else {
					// already a cell; enter
					cursor = c.r[q];
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

	float zz_rec(int i, float x0, float y0, float s, struct vec2& pz) {
		struct cell& c = *cells[i];
		if(c.m0 == 0) return 0;
		c.normalize_z();
		float h = s*0.5f;
		float x0h = x0 + h;
		float y0h = y0 + h;
		float rimax = 0;
		if(h > LS) {
			if(c.r[0]) rimax = max<float>(rimax, zz_rec(c.r[0], x0, y0, h, c.z));
			if(c.r[1]) rimax = max<float>(rimax, zz_rec(c.r[1], x0h, y0, h, c.z));
			if(c.r[2]) rimax = max<float>(rimax, zz_rec(c.r[2], x0, y0h, h, c.z));
			if(c.r[3]) rimax = max<float>(rimax, zz_rec(c.r[3], x0h, y0h, h, c.z));
		}

		struct vec2 d;
		if(c.z.x <= (x0h)) {
			if(c.z.y <= (y0h)) {
				d = vec2(x0+s, y0+s);
			} else {
				d = vec2(x0+s, y0);
			}
		} else {
			if(c.z.y <= (y0h)) {
				d = vec2(x0, y0+s);
			} else {
				d = vec2(x0, y0+s);
			}
		}

		c.rmax = d.distance(c.z);
		if(rimax > 0 && rimax < c.rmax) c.rmax = rimax;
		//printf("s:%f rmax:%f\n", s, c.rmax);
		float dz = c.z.distance(pz);
		return c.rmax + dz;
	}

	void zz() {
		struct vec2 pz;
		zz_rec(0, 0, 0, TS, pz);
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
					int n = 0;
					for(float ly = 0; ly < LS; ly+=1) {
						for(float lx = 0; lx < LS; lx+=1) {
							float m = init[pp++];
							if(m > 0) {
								leaf.masses[lp++].m0 = m;
								zm += m;
								zx += m * (x0 + lx);
								zy += m * (y0 + ly);
								n++;
							}
						}
						pp += TS-LS;
					}
					if(n > 0) {
						pass01(x, y, zm, zx, zy, leaf, n);
					}
					p += LS;
				}
				p += (LS-1) * TS;
			}
		}

		printf("number of masses: %d\n", cells[0]->n);
		printf("number of cells: %d\n", cells.last());
		printf("number of leafs: %d\n", leafs.last());

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

	int XXX_well_separated;
	int XXX_direct_ll;
	int XXX_direct_ll_n;

	void direct_ll(int a_index, float a_x0, float a_y0, int b_index, float b_x0, float b_y0) {
		XXX_direct_ll++;
		float dx = a_x0 - b_x0;
		float dy = a_y0 - b_y0;
		float d = sqrtf(dx*dx + dy*dy);

		if(a_index < 0 || a_index > leafs.last()) {
			printf("leaf index a %d out of bounds (last: %d)!\n", a_index, leafs.last());
			exit(1);
		}
		if(b_index < 0 || b_index > leafs.last()) {
			printf("leaf index b %d out of bounds (last: %d)!\n", b_index, leafs.last());
			exit(1);
		}
		struct leaf& a = *leafs[a_index];
		struct leaf& b = *leafs[b_index];

		int ai = 0;
		for(float ady = 0; ady < LS; ady+=1) {
			for(float adx = 0; adx < LS; adx+=1) {
				struct mass& mass_a = a.masses[ai];
				if(mass_a.m0 > 0) {
					int bi = 0;
					for(float bdy = 0; bdy < LS; bdy+=1) {
						for(float bdx = 0; bdx < LS; bdx+=1) {
							struct mass& mass_b = b.masses[bi];
							if(mass_b.m0 > 0) {
								XXX_direct_ll_n++;
								float ax = a_x0 + adx;
								float ay = a_y0 + ady;
								float bx = b_x0 + bdx;
								float by = b_y0 + bdy;
								float rx = bx - ax;
								float ry = by - ay;
								float r2 = rx*rx + ry*ry;
								float r = sqrtf(r2);
								float d0 = 1/r;
								float d1 = -1/(r2*r);
								float d2 = 2/(r2*r2*r);
								float d3 = -4/(r2*r2*r2*r);
							}
							bi++;
						}
					}
				}
				ai++;
			}
		}
	}

	void direct_cl(int cell_index, float cell_x0, float cell_y0, float cell_size, int leaf_index, float leaf_x0, float leaf_y0) {
		struct cell& cell = *cells[cell_index];
		float h = cell_size * 0.5f;
		if(h > LS) {
			if(cell.r[0]) direct_cl(cell.r[0], cell_x0, cell_y0, h, leaf_index, leaf_x0, leaf_y0);
			if(cell.r[1]) direct_cl(cell.r[1], cell_x0+h, cell_y0, h, leaf_index, leaf_x0, leaf_y0);
			if(cell.r[2]) direct_cl(cell.r[2], cell_x0, cell_y0+h, h, leaf_index, leaf_x0, leaf_y0);
			if(cell.r[3]) direct_cl(cell.r[3], cell_x0+h, cell_y0+h, h, leaf_index, leaf_x0, leaf_y0);
		} else {
			if(cell.r[0]) direct_ll(cell.r[0], cell_x0, cell_y0, leaf_index, leaf_x0, leaf_y0);
			if(cell.r[1]) direct_ll(cell.r[1], cell_x0+h, cell_y0, leaf_index, leaf_x0, leaf_y0);
			if(cell.r[2]) direct_ll(cell.r[2], cell_x0, cell_y0+h, leaf_index, leaf_x0, leaf_y0);
			if(cell.r[3]) direct_ll(cell.r[3], cell_x0+h, cell_y0+h, leaf_index, leaf_x0, leaf_y0);
		}
	}

	void direct_c2(int index, float x0, float y0, float size) {
		struct cell& cell = *cells[index];
		float h = size*0.5f;
		if(h > LS) {
			if(cell.r[0]) direct_c2(cell.r[0], x0, y0, h);
			if(cell.r[1]) direct_c2(cell.r[1], x0+h, y0, h);
			if(cell.r[2]) direct_c2(cell.r[2], x0, y0+h, h);
			if(cell.r[3]) direct_c2(cell.r[3], x0+h, y0+h, h);
			if(cell.r[0] && cell.r[1]) direct_cc(cell.r[0], x0, y0, h, cell.r[1], x0+h, y0, h);
			if(cell.r[2] && cell.r[3]) direct_cc(cell.r[2], x0, y0+h, h, cell.r[3], x0+h, y0+h, h);
			if(cell.r[0] && cell.r[2]) direct_cc(cell.r[0], x0, y0, h, cell.r[2], x0, y0+h, h);
			if(cell.r[1] && cell.r[3]) direct_cc(cell.r[1], x0+h, y0, h, cell.r[3], x0+h, y0+h, h);
			if(cell.r[0] && cell.r[3]) direct_cc(cell.r[0], x0, y0, h, cell.r[3], x0+h, y0+h, h);
			if(cell.r[1] && cell.r[2]) direct_cc(cell.r[1], x0+h, y0, h, cell.r[2], x0, y0+h, h);
		} else {
			if(cell.r[0]) direct_ll(cell.r[0], x0, y0, cell.r[0], x0, y0);
			if(cell.r[1]) direct_ll(cell.r[1], x0+h, y0, cell.r[1], x0+h, y0);
			if(cell.r[2]) direct_ll(cell.r[2], x0, y0+h, cell.r[2], x0, y0+h);
			if(cell.r[3]) direct_ll(cell.r[3], x0+h, y0+h, cell.r[3], x0+h, y0+h);
			if(cell.r[0] && cell.r[1]) direct_ll(cell.r[0], x0, y0, cell.r[1], x0+h, y0);
			if(cell.r[2] && cell.r[3]) direct_ll(cell.r[2], x0, y0+h, cell.r[3], x0+h, y0+h);
			if(cell.r[0] && cell.r[2]) direct_ll(cell.r[0], x0, y0, cell.r[2], x0, y0+h);
			if(cell.r[1] && cell.r[3]) direct_ll(cell.r[1], x0+h, y0, cell.r[3], x0+h, y0+h);
			if(cell.r[0] && cell.r[3]) direct_ll(cell.r[0], x0, y0, cell.r[3], x0+h, y0+h);
			if(cell.r[1] && cell.r[2]) direct_ll(cell.r[1], x0+h, y0, cell.r[2], x0, y0+h);
		}
	}

	void direct_cc(int a_index, float a_x0, float a_y0, float a_size, int b_index, float b_x0, float b_y0, float b_size) {
		struct cell& a = *cells[a_index];
		struct cell& b = *cells[b_index];
		float ha = a_size * 0.5f;
		float hb = b_size * 0.5f;

		if(ha > LS && hb > LS) {
			for(int qa = 0; qa < 4; qa++) {
				for(int qb = 0; qb < 4; qb++) {
					if(a.r[qa] && b.r[qb]) {
						direct_cc(
							a.r[qa],
							a_x0 + ((qa&1)?ha:0),
							a_y0 + ((qa&2)?ha:0),
							ha,
							b.r[qb],
							b_x0 + ((qa&1)?hb:0),
							b_y0 + ((qb&2)?hb:0),
							hb);
					}
				}
			}
		} else if(ha > LS) {
			for(int qa = 0; qa < 4; qa++) {
				for(int qb = 0; qb < 4; qb++) {
					if(a.r[qa] && b.r[qb]) {
						direct_cl(
							a.r[qa],
							a_x0 + ((qa&1)?ha:0),
							a_y0 + ((qa&2)?ha:0),
							ha,
							b.r[qb],
							b_x0 + ((qa&1)?hb:0),
							b_y0 + ((qb&2)?hb:0));
					}
				}
			}
		} else if(hb > LS) {
			for(int qa = 0; qa < 4; qa++) {
				for(int qb = 0; qb < 4; qb++) {
					if(a.r[qa] && b.r[qb]) {
						direct_cl(
							b.r[qb],
							b_x0 + ((qa&1)?hb:0),
							b_y0 + ((qb&2)?hb:0),
							hb,
							a.r[qa],
							a_x0 + ((qa&1)?ha:0),
							a_y0 + ((qa&2)?ha:0));
					}
				}
			}
		} else {
			for(int qa = 0; qa < 4; qa++) {
				for(int qb = 0; qb < 4; qb++) {
					if(a.r[qa] && b.r[qb]) {
						direct_ll(
							b.r[qb],
							b_x0 + ((qa&1)?hb:0),
							b_y0 + ((qb&2)?hb:0),
							a.r[qa],
							a_x0 + ((qa&1)?ha:0),
							a_y0 + ((qa&2)?ha:0));
					}
				}
			}
		}
	}

	void interact_cc(int a_index, float a_x0, float a_y0, float a_size, int b_index, float b_x0, float b_y0, float b_size) {
		if(a_size > LS && b_size > LS) {
			const int Ncs = 64;
			const long Ncc_post = 16;
			const float threshold = 1.0f / 0.65f;

			struct cell& a = *cells[a_index];
			struct cell& b = *cells[b_index];

			// bail if either cell is completely empty
			if(a.n == 0) return;
			if(b.n == 0) return;

			if(a_index == b_index) {
				if(a.n <= Ncs) {
					// direct summation
					//printf("TODO direct summation %d (n=%d  s=%f)\n", a_index, a.n, a_size);
					direct_c2(a_index, a_x0, a_y0, a_size);
				} else {
					// split
					float h = a_size * 0.5f;

					if(a.r[0]) interact_cc(a.r[0], a_x0, a_y0, h, a.r[0], a_x0, a_y0, h);
					if(a.r[1]) interact_cc(a.r[1], a_x0+h, a_y0, h, a.r[1], a_x0+h, a_y0, h);
					if(a.r[2]) interact_cc(a.r[2], a_x0, a_y0+h, h, a.r[2], a_x0, a_y0+h, h);
					if(a.r[3]) interact_cc(a.r[3], a_x0+h, a_y0+h, h, a.r[3], a_x0+h, a_y0+h, h);

					if(a.r[0] && a.r[1]) interact_cc(a.r[0], a_x0, a_y0, h, a.r[1], a_x0+h, a_y0, h);
					if(a.r[2] && a.r[3]) interact_cc(a.r[2], a_x0, a_y0+h, h, a.r[3], a_x0+h, a_y0+h, h);
					if(a.r[0] && a.r[2]) interact_cc(a.r[0], a_x0, a_y0, h, a.r[2], a_x0, a_y0+h, h);
					if(a.r[1] && a.r[3]) interact_cc(a.r[1], a_x0+h, a_y0, h, a.r[3], a_x0+h, a_y0+h, h);
					if(a.r[0] && a.r[3]) interact_cc(a.r[0], a_x0, a_y0, h, a.r[3], a_x0+h, a_y0+h, h);
					if(a.r[1] && a.r[2]) interact_cc(a.r[1], a_x0+h, a_y0, h, a.r[2], a_x0, a_y0+h, h);
				}
			} else if(a.z.distance(b.z) > ((a.rmax + b.rmax) * threshold)) {
				// well separated
				//printf("TODO well separated %d vs %d\n", a_index, b_index);
				XXX_well_separated++;
			} else if((((long)a.n)*((long)b.n)) < Ncc_post) {
				// direct summation
				//printf("TODO direct summation %d (n:%d) vs %d (n:%d)\n", a_index, a.n, b_index, b.n);
				direct_cc(a_index, a_x0, a_y0, a_size, b_index, b_x0, b_y0, b_size);
			} else if(a.rmax > b.rmax) {
				// split a
				float h = a_size * 0.5f;
				if(a.r[0]) interact_cc(a.r[0], a_x0, a_y0, h, b_index, b_x0, b_y0, b_size);
				if(a.r[1]) interact_cc(a.r[1], a_x0+h, a_y0, h, b_index, b_x0, b_y0, b_size);
				if(a.r[2]) interact_cc(a.r[2], a_x0, a_y0+h, h, b_index, b_x0, b_y0, b_size);
				if(a.r[3]) interact_cc(a.r[3], a_x0+h, a_y0+h, h, b_index, b_x0, b_y0, b_size);
			} else {
				// split b
				float h = b_size * 0.5f;
				if(b.r[0]) interact_cc(a_index, a_x0, a_y0, a_size, b.r[0], b_x0, b_y0, h);
				if(b.r[1]) interact_cc(a_index, a_x0, a_y0, a_size, b.r[1], b_x0+h, b_y0, h);
				if(b.r[2]) interact_cc(a_index, a_x0, a_y0, a_size, b.r[2], b_x0, b_y0+h, h);
				if(b.r[3]) interact_cc(a_index, a_x0, a_y0, a_size, b.r[3], b_x0+h, b_y0+h, h);
			}
		} else if(a_size > LS) {
			direct_cl(a_index, a_x0, a_y0, a_size, b_index, b_x0, b_y0);
		} else if(b_size > LS) {
			direct_cl(b_index, b_x0, b_y0, b_size, a_index, a_x0, a_y0);
		} else {
			direct_ll(a_index, a_x0, a_y0, b_index, b_x0, b_y0);
			//printf("TODO bottom %d (size:%f) %d (size:%f)\n", a_index, a_size, b_index, b_size);
		}
	}

	void interact() {
		XXX_well_separated = 0;
		XXX_direct_ll = 0;
		XXX_direct_ll_n = 0;
		{
			scope_timer t0("interact");
			interact_cc(0,0,0,TS,0,0,0,TS);
		}
		printf("well separated cell interactions: %d\n", XXX_well_separated);
		printf("direct leaf-leaf interactions: %d\n", XXX_direct_ll);
		printf("direct mass-mass interactions: %d\n",  XXX_direct_ll_n);
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
	}
	printf("\n\n-------------------\n");
	{
		scope_timer t0("proper");
		t.build2(init);
		t.interact();
		t.evaluate();
	}

	return 0;
}

