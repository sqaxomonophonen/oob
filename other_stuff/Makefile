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

barnes: barnes.o nanotime.o Makefile
	$(CC) $(CCFLAGS) -o barnes barnes.o nanotime.o $(LINK)

dehnen: dehnen.o nanotime.o Makefile
	$(CC) $(CCFLAGS) -o dehnen dehnen.o nanotime.o $(LINK)

conv: conv.o nanotime.o Makefile
	$(CC) $(CCFLAGS) -o conv conv.o nanotime.o $(LINK)

barcon: barcon.o nanotime.o Makefile
	$(CC) $(CCFLAGS) -o barcon barcon.o nanotime.o $(LINK)

clean:
	rm -f $(OWNOBJS) main

cleanall:
	rm -f $(ALLOBJS) main
