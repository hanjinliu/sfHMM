# distutils: language = c++

from libcpp.vector cimport vector
cimport numpy as np

cdef extern from "StepFinder.h" namespace "stepc":
    cdef cppclass Base:
        vector[double] data
        vector[double] fit
        double penalty
        int len
        int n_step
        vector[int] step_list
        vector[int] len_list
        vector[double] mu_list
        vector[double] step_size_list

        Base(vector[double]& DataInput, double p) except +
        void AddInfo()

    cdef cppclass GaussStep(Base):
        vector[double] data
        vector[double] fit
        double penalty
        int len
        int n_step
        vector[int] step_list
        vector[int] len_list
        vector[double] mu_list
        vector[double] step_size_list

        GaussStep(vector[double]& DataInput, double p) except +
        double OneStepFinding(int start, int end, int* pos)
        void AppendSteps()
        void MultiStepFinding()
        void AddInfo()
    
    cdef cppclass PoissonStep(Base):
        vector[double] data
        vector[double] fit
        double penalty
        int len
        int n_step
        vector[int] step_list
        vector[int] len_list
        vector[double] mu_list
        vector[double] step_size_list

        PoissonStep(vector[double]& DataInput, double p) except +
        double OneStepFinding(int start, int end, int* pos)
        void AppendSteps(int start, int end)
        void MultiStepFinding()
        void AddInfo()


cdef class PyGaussStep:
    cdef GaussStep* thisptr

    def __cinit__(self, np.ndarray[np.float64_t, ndim=1] data, double p=0):
        self.thisptr = new GaussStep(list(data), p)

    def __dealloc__(self):
        del self.thisptr

    def multi_step_finding(self):
        return self.thisptr.MultiStepFinding()

    @property 
    def data(self):
        return self.thisptr.data
    
    @data.setter
    def data(self, data_):
        self.thisptr.data = data_
    
    @property
    def fit(self):
        return self.thisptr.fit
    
    @property
    def n_step(self):
        return self.thisptr.n_step

    @property
    def step_list(self):
        return self.thisptr.step_list

    @property
    def mu_list(self):
        return self.thisptr.mu_list

    @property
    def len_list(self):
        return self.thisptr.len_list
    
    @property
    def step_size_list(self):
        return self.thisptr.step_size_list

cdef class PyPoissonStep:
    cdef PoissonStep* thisptr

    def __cinit__(self, np.ndarray data, double p=0):
        if ((data < 0).any()):
            raise ValueError("Raw data cannot contain negative value.")
        self.thisptr = new PoissonStep(list(data.astype("float64")), p)

    def __dealloc__(self):
        del self.thisptr

    def multi_step_finding(self):
        return self.thisptr.MultiStepFinding()
    
    @property 
    def data(self):
        return self.thisptr.data
    
    @data.setter
    def data(self, data_):
        self.thisptr.data = data_
    
    @property
    def fit(self):
        return self.thisptr.fit
    
    @property
    def n_step(self):
        return self.thisptr.n_step

    @property
    def step_list(self):
        return self.thisptr.step_list

    @property
    def mu_list(self):
        return self.thisptr.mu_list

    @property
    def len_list(self):
        return self.thisptr.len_list
    
    @property
    def step_size_list(self):
        return self.thisptr.step_size_list

