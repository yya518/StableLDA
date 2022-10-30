CC      = g++
CFLAGS  = -g
LDFLAGS	= -lm
SRCS	= estimator.cpp nodes.cpp utility.cpp
OBJS	= estimator.o nodes.o utility.o

default: train

train: train.cpp $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(OBJS)

clean:
	#del *.o train.exe
	rm -f *.o main

