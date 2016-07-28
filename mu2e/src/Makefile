CC = clang++

CFLAGS  = -Wall -std=c++11 -lpthread -lgsl

# the build target executable:
OBJECTS = fiteval_c.o main.o

$(TARGET): $(TARGET).cc
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cc

%.o: %.cc
	$(CC) $(CFLAGS) -c $<

main.exe: $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o main.exe

clean:
	$(RM) $(OBJECTS) main.exe
