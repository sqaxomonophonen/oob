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

/* ugly hax for encoding either a 31bit offset or a positive float in 4 bytes,
 * abusing the fact that the most significant bit of a single-precision float
 * is its sign */
struct enc {
	union vt {
		float mass;
		unsigned int index;
	} v;

	enc() { v.index = 0; }

	bool is_empty() const {
		return v.index == 0;
	}

	void set_index(unsigned int i) {
		v.index = i << 1;
	}

	unsigned int get_index() const {
		return v.index >> 1;
	}

	bool is_mass() const {
		return v.index&1;
	}

	bool is_index() const {
		return !is_mass();
	}

	void set_mass(float m) {
		v.mass = m;
		v.index <<= 1;
		v.index |= 1;
	}

	float get_mass() const {
		union vt _v = v;
		_v.index >>= 1;
		return _v.mass;
	}
};

struct node {
	struct enc r[4];
	float zmx, zmy, zm;

	void add_mass(float x, float y, float m) {
		zmx += x*m;
		zmy += y*m;
		zm += m;
	}
};

struct barnes_hut {
	struct lst<struct node> nodes;

	struct node empty_node;

	barnes_hut() {
		for(int i = 0; i < 4; i++) empty_node.r[i].v.index = 0;
		empty_node.zmx = 0;
		empty_node.zmy = 0;
		empty_node.zm = 0;
	}

	void reset() {
		nodes.reset();
		nodes.write(&empty_node);
	}

	void add_mass(int x, int y, float m) {
		unsigned int cursor = 0;
		float fx = x;
		float fy = y;
		int mask = (1<<(X-1))-1;
		for(int i = (X-1); i >= 0; i--) {
			struct node& n = *nodes[cursor];
			n.add_mass(fx, fy, m);
			int q = ((x>>i)?1:0)+((y>>i)?2:0);
			struct enc& e = n.r[q];
			if(e.is_empty()) {
				// empty; make a node
				nodes.write(&empty_node);
				int new_cursor = nodes.last();
				nodes[cursor]->r[q].set_index(new_cursor);
				cursor = new_cursor;
			} else {
				// already a node; recurse
				cursor = e.get_index();
			}
			x &= mask;
			y &= mask;
			mask >>= 1;
		}
	}

	void update_center_of_mass() {
		for(int i = 0; i < nodes.size; i++) {
			struct node& n = *nodes[i];
			n.zmx /= n.zm;
			n.zmy /= n.zm;
		}
	}

	void build(float* init) {
		reset();
		int ptr = 0;
		int nmass = 0;
		for(int y = 0; y < S; y++) {
			for(int x = 0; x < S; x++) {
				float m = init[ptr++];
				if(m > 0.0f) {
					add_mass(x, y, m);
					nmass++;
				}
			}
		}
		printf("number of nodes: %d / number of non-zero masses: %d\n", nodes.last(), nmass);
	}

	struct cell_grav_stack_element {
		int index;
		int level;
		float width_sqr;
		cell_grav_stack_element() : index(0), level(0), width_sqr(0) {}
		cell_grav_stack_element(int index, int level, float width_sqr) : index(index), level(level), width_sqr(width_sqr) {}
	};

	struct {
		struct cell_grav_stack_element elements[1024];
		int top;

		void push(const struct cell_grav_stack_element& e) {
			top++;
			elements[top] = e;
		}

		struct cell_grav_stack_element& pop() {
			return elements[top--];
		}

	} cell_grav_stack;

	void reset_cell_grav_stack() {
		cell_grav_stack.top = -1;
	}

	void cell_grav(int x, int y, float& ax, float& ay) {
		const float threshold = 0.5;
		const float threshold_sqr = threshold * threshold;
		float fx = x;
		float fy = y;
		ax = 0;
		ay = 0;

		cell_grav_stack.push(cell_grav_stack_element(0, X, (float)(S*S)));

		while(cell_grav_stack.top >= 0) {
			struct cell_grav_stack_element& e = cell_grav_stack.pop();
			struct node& n = *nodes[e.index];
			float dx = n.zmx - fx;
			float dy = n.zmy - fy;
			float dsqr = dx*dx + dy*dy;
			if(dsqr < 0.001f) continue;
			if(e.level <= 1 || (e.width_sqr / dsqr) < threshold_sqr) {
				// below threshold; add acceleration and
				// recurse no further
				float s = (n.zm/dsqr)/sqrtf(dsqr);
				ax += dx*s;
				ay += dy*s;
			} else {
				// above threshold; recurse deeper..
				// XXX stop recursion here!
				float deeper_width_sqr = e.width_sqr * 0.25f;
				int deeper_level = e.level - 1;
				for(int q = 0; q < 4; q++) {
					cell_grav_stack.push(cell_grav_stack_element(n.r[q].get_index(), deeper_level, deeper_width_sqr));
				}
			}
		}
	}

	void grav(float* init) {
		reset_cell_grav_stack();
		int ptr = 0;
		float sax = 0;
		float say = 0;
		const int SKIP = 4;
		for(int y = 0; y < S; y+=SKIP) {
			for(int x = 0; x < S; x+=SKIP) {
				float m = init[ptr+=SKIP];
				if(m > 0.0f) {
					float ax, ay;
					cell_grav(x, y, ax, ay);
					sax += ax;
					say += ay;
				}
			}
			ptr+=S*(SKIP-1);
		}
		printf("acc sum: (%f,%f)\n", sax, say);
	}
};

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

	barnes_hut bh;
	{
		scope_timer T("barnes-hut build (1)");
		bh.build(init);
	}
	printf("build warm-up done\n\n\n");
	{
		scope_timer T1("grav step total");
		{
			scope_timer T("barnes-hut build (2)");
			bh.build(init);
		}
		{
			scope_timer T("update center of mass");
			bh.update_center_of_mass();
		}
		{
			scope_timer T("gravity");
			bh.grav(init);
		}
	}
	return 0;
}
