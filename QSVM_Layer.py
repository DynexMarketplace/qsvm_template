import os
import numpy as np
import dimod
from tqdm import tqdm
import logging
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import dynex
import dimod
import neal
import pickle
import time
import tensorflow as tf

logging.basicConfig(filename="QSVM.log", level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class QSVM_Layer(tf.keras.layers.Layer):
    
    def __init__(self, B:int,K:int,C:int,gamma:int,xi:float,dataset,train_percent,sampler_type,mainnet,num_reads,annealing_time):
        """
        This function defines the class of quantum support vector machine.

        Parameters
        ----------
        B, K, C, gamma, xi: SVM model parameters
        dataset: dataset for train and test
        train_percent: the percentage of dataset for training 
        sampler_type: sampler type
                        "DNX" The Dynex Neuromorphic sampler
                        "EXACT" A brute force exact solver which tries all combinations. Very limited problem size
                        "QPU" D-Wave Quantum Processor (QPU) based D-Wave sampler
                        "HQPU" D-Wave Advantage Hybrid Solver
                        "SA" Simulated Annealing using the SimulatedAnnealerSampler from the D-Wave Ocean SDK
        mainnet: use mainnet or not
        num_reads: the number of reads for sampler
        annealing_time: annealing time in DYNEX platform
        """
        super(QSVM_Layer, self).__init__()
        self.B = B
        self.K = K 
        self.C = C 
        self.gamma = gamma
        self.xi = xi 
        self.N = int(len(dataset.data)*train_percent)
        self.sampler_type = sampler_type
        self.data = dataset.data 
        self.t = dataset.target
        self.mainnet = mainnet
        self.num_reads = num_reads
        self.annealing_time = annealing_time

        if(sampler_type == 'HQPU'):
            self.sampler = LeapHybridSampler()
        if(sampler_type == 'SA'):
            self.sampler = neal.SimulatedAnnealingSampler()
        if(sampler_type == 'QPU'):
            self.sampler = EmbeddingComposite(DWaveSampler())
        if(sampler_type == 'DNX'):
            self.sampler = ''
        if(sampler_type == 'EXACT'):
            self.sampler = dimod.ExactSolver()
  
        self.debugging = False
        self.logging = False

        # Log the initialization
        logger.info("Initialized QSVM")
    
    def delta(self, i, j):
        if i == j:
            return 1
        else:
            return 0

    def kernel(self,x, y):
        if self.gamma == -1:
            k = np.dot(x, y)
        elif self.gamma >= 0:
            k = np.exp(-self.gamma*(np.linalg.norm(x-y, ord=2)))
        return k
    
    def call(self,x):
        if tf.is_tensor(x):
            x = x.numpy()
        N = len(self.alpha)
        f = sum([self.alpha[n]*self.t[n]*self.kernel(self.data[n], x) for n in range(self.N)]) + self.b
        logger.debug("Completed forward pass")
        return f
    
        
    def train(self,save_model=True, save_path='./models', model_file='QSVM.model'):
        """
        train the SVM model.

        Parameters:
            - save_model: save the model's state after training.
            - save_path: the path of the model saved.
        """
        Q_tilde = np.zeros((self.K*self.N, self.K*self.N))
        for n in range(self.N):
            for m in range(self.N):
                for k in range(self.K):
                    for j in range(self.K):
                        Q_tilde[(self.K*n+k, self.K*m+j)] = 0.5*(self.B**(k+j))*self.t[n]*self.t[m]*(self.kernel(self.data[n], self.data[m])+self.xi)-(self.delta(n, m)*self.delta(k, j)*(self.B**k))

        Q = np.zeros((self.K*self.N, self.K*self.N))
        for j in range(self.K*self.N):
            Q[(j, j)] = Q_tilde[(j, j)]
            for i in range(self.K*self.N):
                if i < j:
                    Q[(i, j)] = Q_tilde[(i, j)] + Q_tilde[(j, i)]

        size_of_q = Q.shape[0]
        qubo = {(i, j): Q[i, j] for i, j in product(range(size_of_q), range(size_of_q))}

        now = time.perf_counter();
        
        if(self.sampler_type == 'HQPU'):
            response = self.sampler.sample_qubo(qubo)
        if(self.sampler_type == 'SA'):
            response = self.sampler.sample_qubo(qubo, num_reads=self.num_reads)
        if(self.sampler_type == 'QPU'):
            response = self.sampler.sample_qubo(qubo, num_reads=self.num_reads)
        if(self.sampler_type == 'EXACT'):
            response = self.sampler.sample_qubo(qubo)
        if(self.sampler_type == 'DNX'):
            bqm = dimod.BinaryQuadraticModel.from_qubo(qubo);
            model = dynex.BQM(bqm);
            sampler = dynex.DynexSampler(model, mainnet=self.mainnet, description='QSVM');
            response = sampler.sample(num_reads=self.num_reads, annealing_time=self.annealing_time, debugging=False);
            
        print(f'Solver Time: {time.perf_counter() - now}') 

        a = response.first.sample

        self.alpha = []
        for n in range(self.N):
            self.alpha.append(sum([(self.B**k)*a[self.K*n+k] for k in range(self.K)]))

        self.b = sum([self.alpha[n]*(self.C-self.alpha[n])*(self.t[n]-(sum([self.alpha[m]*self.t[m]*self.kernel(self.data[m], self.data[n])
                                                    for m in range(self.N)]))) for n in range(self.N)])/sum([self.alpha[n]*(self.C-self.alpha[n]) for n in range(self.N)])

        
        # Saving the model if specified
        if save_model:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model_save_file = os.path.join(save_path, model_file)
            self.save_model(model_save_file)
            logger.info(f"Model saved at {model_save_file}")
                
        logger.info("Training completed")
        
        
    def save_model(self, file):
        """
        Save the trained model.

        Parameters:
            - path (str): Path to save the model's state.
        """
        checkpoint = {'B':self.B,'K':self.K,'C':self.C,'gamma':self.gamma,'xi':self.xi,'alpha': self.alpha, 'b': self.b}
        with open(file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
    def load_model(self, file):
        """
        Load the model from a saved state.
        
        Parameters:
            - path (str): Path from where to load the model's state.
        """
        with open(file,"rb") as f:
            checkpoint = pickle.load(f)
        self.B = checkpoint['B']
        self.K = checkpoint['K'] 
        self.C = checkpoint['C'] 
        self.gamma = checkpoint['gamma']
        self.xi = checkpoint['xi'] 
        self.alpha = checkpoint['alpha']
        self.b = checkpoint['b']