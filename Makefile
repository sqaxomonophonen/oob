# Makefile

CC = clang++
PKGS=fftw3f sdl2
CCFLAGS = -O3 -msse2 -ffast-math -Wall $(shell pkg-config --cflags $(PKGS))
LINK = $(shell pkg-config --libs $(PKGS))
O=.o
OWNOBJS = main.o nanotime.o
ALLOBJS = $(OWNOBJS)
DEPS = nanotime.h video_out.h

all: main

%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CCFLAGS)

main: $(ALLOBJS) Makefile
	$(CC) $(CCFLAGS) -o main \
		$(ALLOBJS) \
		$(LINK)

clean:
	rm -f $(OWNOBJS) main

cleanall:
	rm -f $(ALLOBJS) main
