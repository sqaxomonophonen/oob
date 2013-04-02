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

struct node {
	unsigned int r[4]; // children
	float mass, zmx, zmy;
	int zn;

	void clear() {
		r[0] = r[1] = r[2] = r[3] = 0;
		mass = zmx = zmy = 0.0f;
	}
	
	void update(float _mass, float _zmx, float _zmy, int _zn) {
		mass += _mass;
		zmx += _zmx;
		zmy += _zmy;
		zn += _zn;
	}

	void normalize() {
		if(zn > 0) {
			zmx /= mass;
			zmy /= mass;
		}
	}
};

struct leaf {
	fftwf_complex masses[LN];
	int n;

	void clear() {
		for(int i = 0; i < LN; i++) {
			masses[i][0] = masses[i][1] = 0.0f;
		}
		n = 0;
	}
};

struct tree {
	struct lst<struct node> nodes;
	struct lst<struct leaf> leafs;

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

	void add_leaf(int x, int y, float mass, float zmx, float zmy, int zn, struct leaf& leaf) {
		unsigned int cursor = 0;
		int mask = (1<<(DX-1))-1;
		for(int i = (DX-1); i >= 0; i--) {
			struct node& n = *nodes[cursor];
			n.update(mass, zmx, zmy, zn);
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
					leaf.n = n;
					add_leaf(x, y, mass, zmx, zmy, n, leaf);
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

	void find_neighbour_convolutions() {
		int nc1x1 = 0;
		int nc1x2 = 0;
		int nc2x2 = 0;
		int ndirect = 0;

		// convolution costs
		#if LX == 3
		const float Cc1x1 = 0.0239f;
		const float Cc1x2 = 0.0509f;
		const float Cc2x2 = 0.114f;
		#elif LX == 4
		const float Cc1x1 = 0.114f;
		const float Cc1x2 = 0.300f;
		const float Cc2x2 = 0.645f;
		#elif LX == 5
		const float Cc1x1 = 0.645f;
		const float Cc1x2 = 1.45f;
		const float Cc2x2 = 3.11f;
		#else
		#error "no convolution costs";
		#endif
		const float Cdirect = Cc1x1 / 2.0f; // XXX GUESS

		// threshold for direct summation (chosen: N)
		const int direct_threshold = LS * LS;

		float Ctotal = 0.0f;
		float Csimple = 0.0f;

		int neighbours[9];
		int nscheme[5] = {0,0,0,0,0};

		for(int y = 0; y < DS; y++) {
			for(int x = 0; x < DS; x++) {
				int l = find_leaf(x, y);
				if(l > 0) {
					Csimple += Cc2x2;
					int ni = 0;
					int n = 0;
					for(int dy = -1; dy <= 1; dy++) {
						for(int dx = -1; dx <= 1; dx++) {
							if(dx == 0 && dy == 0) { 
								neighbours[ni] = l;
							} else {
								int l2 = find_leaf(x + dx, y + dy);
								neighbours[ni] = l2;
								if(l2) n++;
							}
							ni++;
						}
					}
					if(n == 0) {
						// scheme 0
						int ln = leafs[l]->n;
						int ln2 = ln * ln;
						if(ln2 <= direct_threshold) {
							ndirect++;
							Ctotal += Cdirect * (((float)ln2) / ((float)direct_threshold));
						} else {
							nc1x1++;
							Ctotal += Cc1x1;
						}
						nscheme[0]++;
					} else {
						int ln = leafs[l]->n;

						// evaluate scheme 1
						int nc1x1_sc1 = 0;
						int ndirect_sc1 = 0;
						float C_sc1 = 0.0f;
						int scheme = 1;
						{
							int ni = 0;
							for(int dy = -1; dy <= 1; dy++) {
								for(int dx = -1; dx <= 1; dx++) {
									int ln2 = leafs[neighbours[ni++]]->n;
									if(ln2 > 0) {
										ln2 = ln * ln2;
										if(ln2 <= direct_threshold) {
											ndirect_sc1++;
											C_sc1 += Cdirect * (((float)ln2) / ((float)direct_threshold));
										} else {
											nc1x1_sc1++;
											C_sc1 += Cc1x1;
										}
									}
								}
							}
						}
						float Cmin = C_sc1;

						// evaluate scheme 2
						int nc1x1_sc2 = 0;
						int nc1x2_sc2 = 0;
						int ndirect_sc2 = 0;
						float C_sc2 = 0.0f;
						{
							int ni = 0;
							for(int dy = -1; dy <= 1; dy++) {
								float Cs = 0.0f;
								int _n1 = 0;
								int _nd = 0;
								for(int dx = -1; dx <= 1; dx++) {
									int ln2 = leafs[neighbours[ni++]]->n;
									if(ln2 > 0) {
										ln2 = ln * ln2;
										if(ln2 <= direct_threshold) {
											_nd++;
											Cs += Cdirect * (((float)ln2) / ((float)direct_threshold));
										} else {
											_n1++;
											Cs += Cc1x1;
										}
									}
								}
								if(Cs < Cc1x2) {
									C_sc2 += Cs;
									nc1x1_sc2 += _n1;
									ndirect_sc2 += _nd;
								} else {
									C_sc2 += Cc1x2;
									nc1x2_sc2++;
								}
							}
						}
						if(C_sc2 < Cmin) {
							scheme = 2;
							Cmin = C_sc2;
						}

						// evaluate scheme 3
						int nc1x1_sc3 = 0;
						int nc1x2_sc3 = 0;
						int ndirect_sc3 = 0;
						float C_sc3 = 0.0f;
						{
							int ni = 0;
							for(int dx = -1; dx <= 1; dx++) {
								float Cs = 0.0f;
								int _n1 = 0;
								int _nd = 0;
								for(int dy = -1; dy <= 1; dy++) {
									int ln2 = leafs[neighbours[ni]]->n;
									ni += 3;
									if(ln2 > 0) {
										ln2 = ln * ln2;
										if(ln2 <= direct_threshold) {
											_nd++;
											Cs += Cdirect * (((float)ln2) / ((float)direct_threshold));
										} else {
											_n1++;
											Cs += Cc1x1;
										}
									}
								}
								ni -= 8;
								if(Cs < Cc1x2) {
									C_sc3 += Cs;
									nc1x1_sc3 += _n1;
									ndirect_sc3 += _nd;
								} else {
									C_sc3 += Cc1x2;
									nc1x2_sc3++;
								}
							}
						}
						if(C_sc3 < Cmin) {
							scheme = 3;
							Cmin = C_sc3;
						}

						// evaluate scheme 4
						if(Cc2x2 < Cmin) {
							// soooo easy!
							scheme = 4;
							Cmin = Cc2x2;
						}

						Ctotal += Cmin;

						switch(scheme) {
							case 1:
								nscheme[1]++;
								ndirect += ndirect_sc1;
								nc1x1 += nc1x1_sc1;
								break;
							case 2:
								nscheme[2]++;
								ndirect += ndirect_sc2;
								nc1x1 += nc1x1_sc2;
								nc1x2 += nc1x2_sc2;
								break;
							case 3:
								nscheme[3]++;
								ndirect += ndirect_sc3;
								nc1x1 += nc1x1_sc3;
								nc1x2 += nc1x2_sc3;
								break;
							case 4:
								nscheme[4]++;
								nc2x2++;
								break;
							default:
								fprintf(stderr, "unhandled scheme %d\n", scheme);
								exit(1);
								break;
						}
					}
				}
			}
		}

		printf(" nc1x1: %d\n", nc1x1);
		printf(" nc2x2: %d\n", nc2x2);
		printf(" nc1x2: %d\n", nc1x2);
		printf(" ndirect: %d\n", ndirect);
		for(int i = 0; i < 5; i++) {
			printf("  nscheme[%d]: %d\n", i, nscheme[i]);
		}
		printf(" Ctotal: %f\n", Ctotal);
		printf(" Cc2x2: %f\n", Csimple);
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
			scope_timer t0("find neighbour convolutions");
			t.find_neighbour_convolutions();
		}
	}

	return 0;
}
