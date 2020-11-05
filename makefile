
CXX=g++
CFLGAS = 

### ### ### ### ### ### ### ### ### ##
##
## Dipoles

GSLDIR=/home/diogenes1991/gsl-2.6

ifdef GSLDIR
CFLGAS+=-static
COMMON_INC+=-I$(GSLDIR)
COMMON_LIB_DIRS+=-L$(GSLDIR)
COMMON_LIBS+= -lgsl -lgslcblas
endif

TEST_OBJ=$(patsubst %.cpp,%.o,$(wildcard $(shell pwd)/*.cpp))

##
## End of Dipoles 
##
### ### ### ### ### ### ### ### ### ##


all: Test

Test: $(TEST_OBJ)
	@echo $(CXX) DNN  
	$(CXX) $(CFLAGS) -o DNN $(COMMON_INC) $(COMMON_LIB_DIRS) -L. $(INTEGRAND_OBJ) DNN.o $(COMMON_LIBS)

%.o:%.cpp
	@echo $(CXX) -c $< 
	@$(CXX) -c $(CFLAGS) $< -o $@ $(COMMON_INC) $(COMMON_LIB_DIRS) $(COMMON_LIBS)

clean:
	@rm -f *.o