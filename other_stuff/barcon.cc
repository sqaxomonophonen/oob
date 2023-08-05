#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <fftw3.h>

#include "nanotime.h"

#define TX (9)
#define TS (1<<TX)
#define TN (1<<(TX*2))

#define LX (3)
#define LS (1<<LX)
#define LN (1<<(LX*2))

#define DX (TX-LX)
#define DS (1<<(DX))
#define DN (1<<(DX*2))

#define ZX (LX+3)
#define ZS (1<<(ZX))
#define ZN (1<<(ZX*2))

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

struct leaf {
	fftwf_complex masses[LN];
	float mass, zmx, zmy;

	void clear() {
		bzero(masses, LN * sizeof(fftwf_complex));
		mass = zmx = zmy = 0.0f;
	}

	void normalize() {
		zmx /= mass;
		zmy /= mass;
	}
};

struct node {
	unsigned int r[4]; // children
	float mass, zmx, zmy;
	int zn;

	void clear() {
		r[0] = r[1] = r[2] = r[3] = 0;
		mass = zmx = zmy = 0.0f;
		zn = 0;
	}

	void update(struct leaf& leaf, int _zn) {
		mass += leaf.mass;
		zmx += leaf.zmx;
		zmy += leaf.zmy;
		zn += _zn;
	}

	void normalize() {
		if(!is_empty()) {
			zmx /= mass;
			zmy /= mass;
		}
	}

	bool is_empty() const {
		return zn == 0;
	}
};

struct gravity_convolution {
	fftwf_plan data_forward_plan;
	fftwf_plan data_backward_plan;
	fftwf_complex *placeholder;
	fftwf_complex *kernel;

	gravity_convolution() {
		kernel = (fftwf_complex*) fftwf_malloc(ZN * sizeof(fftwf_complex));

		const int nz = ZS>>1;
		const int mid = nz>>1;

		fftwf_plan kernel_plan = fftwf_plan_dft_2d(ZS, ZS, kernel, kernel, -1, 0);
		for(int y = 0; y < ZS; y++) {
			for(int x = 0; x < ZS; x++) {
				if(x < nz && y < nz && !(x == mid && y == mid)) {
					float dx = mid - x;
					float dy = mid - y;
					float d2 = dx*dx + dy*dy;
					float di = 1.0f / sqrtf(d2);
					kernel_at(x, y)[0] = (dx * di) / d2;
					kernel_at(x, y)[1] = (dy * di) / d2;
				} else {
					// zero padding
					kernel_at(x, y)[0] = 0.0f;
					kernel_at(x, y)[1] = 0.0f;
				}
			}
		}
		fftwf_execute(kernel_plan);

		placeholder = (fftwf_complex*) fftwf_malloc(ZN * sizeof(fftwf_complex));
		data_forward_plan = fftwf_plan_dft_2d(ZS, ZS, placeholder, placeholder, -1, 0);
		data_backward_plan = fftwf_plan_dft_2d(ZS, ZS, placeholder, placeholder, 1, 0);
	}

	~gravity_convolution() {
		fftwf_free(placeholder);
		fftwf_free(kernel);
	}

	fftwf_complex& kernel_at(int x, int y) const {
		return kernel[x + (y << ZX)];
	}

	void convolve(fftwf_complex* z) {
		fftwf_execute_dft(data_forward_plan, z, z);
		for(int i = 0; i < ZN; i++) {
			float new_re = z[i][0] * kernel[i][0] - z[i][1] * kernel[i][1];
			z[i][1] = z[i][1] * kernel[i][0] + z[i][0] * kernel[i][1];
			z[i][0] = new_re;
		}
		fftwf_execute_dft(data_backward_plan, z, z);
	}
};

struct tree {
	struct gravity_convolution gc;

	struct lst<struct node> nodes;
	struct lst<struct leaf> leafs;
	struct lst<fftwf_complex*> zs;

	struct node empty_node;
	struct leaf empty_leaf;

	tree() {
		empty_node.clear();
		empty_leaf.clear();
	}

	void reset() {
		nodes.reset();
		nodes.write(&empty_node);
		leafs.reset();
		leafs.write(&empty_leaf);
	}

	void add_leaf(int x, int y, int zn, struct leaf& leaf) {
		unsigned int cursor = 0;
		int mask = (1<<(DX-1))-1;
		for(int i = (DX-1); i >= 0; i--) {
			struct node& n = *nodes[cursor];
			n.update(leaf, zn);
			int q = ((x>>i)?1:0)+((y>>i)?2:0);
			if(i > 0) {
				if(n.r[q] == 0) {
					// empty; make new cell
					nodes.write(&empty_node);
					int new_cursor = nodes.last();
					nodes[cursor]->r[q] = new_cursor;
					cursor = new_cursor;
				} else {
					// already a cell; enter
					cursor = n.r[q];
				}
			} else {
				leafs.write(&leaf);
				nodes[cursor]->r[q] = leafs.last();
				return;
			}
			x &= mask;
			y &= mask;
			mask >>= 1;
		}
	}

	void add_zleaf() {
		while(zs.last() < (leafs.last()-1)) {
			fftwf_complex* f = (fftwf_complex*) fftwf_malloc(ZN * sizeof(fftwf_complex));
			zs.write(&f);
		}
		struct leaf& leaf = *leafs.top();
		fftwf_complex* f = *zs.top();

		bzero(f, ZN * sizeof(fftwf_complex));

		int i = 0;
		int j = 0;
		for(int y = 0; y < LS; y++) {
			for(int x = 0; x < LS; x++) {
				f[j][0] = leaf.masses[i][0];
				f[j][1] = leaf.masses[i][1];
				i++;
				j++;
			}
			j += ZS-LS;
		}
	}

	void build(float* init) {
		reset();
		int p = 0;
		for(int y = 0; y < DS; y++) {
			for(int x = 0; x < DS; x++) {
				struct leaf leaf;
				leaf.clear();
				float zmx = 0.0f;
				float zmy = 0.0f;
				float mass = 0.0f;
				int pp = p;
				int lp = 0;
				float x0 = x * LS;
				float y0 = y * LS;
				int n = 0;
				for(float ly = 0; ly < LS; ly+=1) {
					for(float lx = 0; lx < LS; lx+=1) {
						float m = init[pp++];
						if(m > 0) {
							leaf.masses[lp][0] = m;
							leaf.masses[lp][1] = 0.0f;
							mass += m;
							zmx += m * (x0 + lx);
							zmy += m * (y0 + ly);
							n++;
						}
						lp++;
					}
				}
				if(n > 0) {
					leaf.mass = mass;
					leaf.zmx = zmx;
					leaf.zmy = zmy;
					add_leaf(x, y, n, leaf);
					add_zleaf();
				}
				p += LS;
			}
			p += (LS-1) * TS;
		}
		printf(" nodes: %d\n", nodes.last());
		printf(" leafs: %d\n", leafs.last());
	}

	void normalize() {
		for(int i = 0; i <= nodes.last(); i++) {
			struct node& n = *nodes[i];
			n.normalize();
		}
		for(int i = 0; i <= leafs.last(); i++) {
			struct leaf& l = *leafs[i];
			l.normalize();
		}
	}

	void convolve() {
		for(int i = 0; i <= (leafs.last()-1); i++) {
			gc.convolve(*zs[i]);
		}
	}

	int find_leaf(int x, int y) {
		if(x < 0 || y < 0 || x >= DS || y >= DS) return 0;
		unsigned int cursor = 0;
		int mask = (1<<(DX-1))-1;
		for(int i = (DX-1); i >= 0; i--) {
			struct node& n = *nodes[cursor];
			int q = ((x>>i)?1:0)+((y>>i)?2:0);
			if(i > 0) {
				if(n.r[q] == 0) return 0;
				cursor = n.r[q];
			} else {
				return nodes[cursor]->r[q];
			}
			x &= mask;
			y &= mask;
			mask >>= 1;
		}
		return 0;
	}

	void barnes_rec(float x0, float y0, int n, int d, fftwf_complex& res) {
		const float threshold = 0.5;
		const float threshold_sqr = threshold * threshold;

		struct node& node = *nodes[n];

		float dx = node.zmx - x0;
		float dy = node.zmy - y0;
		float dsqr = dx*dx + dy*dy;
		float wsqr = 1<<((TX-d)*2);

		if(dsqr < 0.001f) return;

		if(d == (DX-1)) {
			for(int q = 0; q < 4; q++) {
				int r = node.r[q];
				if(r) {
					struct leaf& leaf = *leafs[r];
					float dx1 = leaf.zmx - x0;
					float dy1 = leaf.zmy - y0;
					float dsqr1 = dx1*dx1 + dy1*dy1;
					if(dsqr1 >= 0.001f) {
						float s = (leaf.mass/dsqr1)/sqrtf(dsqr1);
						res[0] += dx1*s;
						res[1] += dy1*s;
					}
				}
			}
		} else if((wsqr / dsqr) < threshold_sqr) {
			float s = (node.mass/dsqr)/sqrtf(dsqr);
			res[0] += dx*s;
			res[1] += dy*s;
		} else {
			for(int q = 0; q < 4; q++) {
				int r = node.r[q];
				if(r) {
					barnes_rec(x0, y0, r, d+1, res);
				}
			}
		}
	}

	void barnes(float x0, float y0, fftwf_complex& res) {
		res[0] = 0;
		res[1] = 0;
		barnes_rec(x0, y0, 0, 0, res);
	}

	void apply_grav() {
		const int mid = ZS>>2;
		for(int y = 0; y < DS; y++) {
			for(int x = 0; x < DS; x++) {
				int l0 = find_leaf(x, y);
				struct leaf& leaf0 = *leafs[l0];
				if(l0) {
					fftwf_complex init;
					barnes(leaf0.zmx, leaf0.zmy, init);
					{
						int i0 = 0;
						for(int ly = 0; ly < LS; ly++) {
							for(int lx = 0; lx < LS; lx++) {
								leaf0.masses[i0][0] = init[0];
								leaf0.masses[i0][1] = init[1];
								i0++;
							}
						}
					}

					for(int dy = -1; dy <= 1; dy++) {
						for(int dx = -1; dx <= 1; dx++) {
							int l1 = dx == 0 && dy == 0 ? l0 : find_leaf(x+dx, y+dy);
							if(l1) {
								fftwf_complex* z1 = *zs[l1-1];
								int i0 = 0;
								int i1 = mid-dx*LS + ((mid-dy*LS)<<ZX);
								for(int ly = 0; ly < LS; ly++) {
									for(int lx = 0; lx < LS; lx++) {
										leaf0.masses[i0][0] += z1[i1][0];
										leaf0.masses[i0][1] += z1[i1][1];
										i0++;
										i1++;
									}
									i1 += ZS-LS;
								}
							}
						}
					}
				}
			}
		}
	}

};

int main(int argc, char** argv) {
	if(argc != 2) {
		fprintf(stderr, "usage: %s <%dx%d.gray>\n", argv[0], TS, TS);
		exit(1);
	}

	char* initc = (char*) malloc(TN);

	{
		int fd = open(argv[1], O_RDONLY);
		readn(fd, initc, TN);
		close(fd);
	}

	float* init = (float*) malloc(TN * sizeof(float));
	for(int i = 0; i < TN; i++) init[i] = (float) initc[i];

	printf("world: %dx%d\n", TS, TS);
	printf("leaf: %dx%d\n", LS, LS);
	printf("world leafs: %dx%d\n\n", DS, DS);

	struct tree t;
	{
		scope_timer t0("warm-up");
		t.build(init);
	}
	printf("\n\n");
	{
		scope_timer t1("TOTAL");
		{
			scope_timer t0("build");
			t.build(init);
		}
		{
			scope_timer t0("normalize");
			t.normalize();
		}
		{
			scope_timer t0("convolve");
			t.convolve();
		}
		{
			scope_timer t0("apply gravity");
			t.apply_grav();
		}
	}

	return 0;
}
