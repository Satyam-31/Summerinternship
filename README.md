# Quantum Algorithms and subroutines

Following are few quantum algorithms as well as the qiskit implementation

The following packages and imports will be used for all the following programs:

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_qsphere
import numpy as np
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit,  execute
from qiskit.tools.jupyter import *
provider=IBMQ.load_account()
```

The following command is used for running on simulator
 and for real system we just choose appropriate backend:
```python
simulator=Aer.get_backend('qasm_simulator')
result=execute(circuit,backend=simulator,shots=1000).result()
counts=result.get_counts()
from qiskit.tools.visualization import plot_histogram
plot_histogram(counts)
```




## Quantum Teleportation

### Overview
Alice wants to send quantum information to Bob. Specifically, suppose she wants to send the qubit state . This entails passing on informationto Bob.

There exists a theorem in quantum mechanics which states that you cannot simply make an exact copy of an unknown quantum state . This is known as the no-cloning theorem. As a result of this we can see that Alice can't simply generate a copy of  and give the copy to Bob. We can only copy classical states (not superpositions).

However, by taking advantage of two classical bits and an entangled qubit pair, Alice can transfer her state  to Bob. We call this teleportation because, at the end, Bob will have  and Alice won't anymore.


### Implementation

```python
from qiskit import *
circuit=QuantumCircuit(3,3)

#creating initial state
circuit.x(0)
circuit.barrier()

#creating bell pair
circuit.h(1)
circuit.cx(1,2)

#operations by alice
circuit.cx(0,1)
circuit.h(0)

circuit.measure([0,1],[0,1])

#operation by bob
circuit.cz(0,2)
circuit.cx(1,2)

circuit.measure(2,2)
circuit.draw()

```

## Deutsch Josza Algorithms

### Overview
The Deutsch-Jozsa algorithm was the first example of a quantum algorithm that performs better than the best classical algorithm. It showed that there can be advantages to using a quantum computer as a computational tool for a specific problem.

We are given a hidden Boolean function , which takes as input a string of bits, and returns either 0 or 1
The property of the given Boolean function is that it is guaranteed to either be balanced or constant. A constant function returns all 0's or all 1's for any input, while a balanced function returns 0's for exactly half of all inputs and 1's for the other half. Our task is to determine whether the given function is balanced or constant.

Note that the Deutsch-Jozsa problem is an -bit extension of the single bit Deutsch problem.

### Implementation

```python
#First we create a sample oracle .
def dj_oracle(case,n):
    oracle_qc=QuantumCircuit(n+1)
    if case=="balanced":
        for qubit in range(n):
            oracle_qc.cx(qubit,n)
    if case=="constant":
        output=np.random.randint(2)
        if output==1:
            oracle_qc.x(n)
 
    oracle_gate=oracle_qc.to_gate()
    oracle_gate.name="Oracle"
    return oracle_gate

#Then we define the algorithm as a function
def dj_algorithm(n,case='random'):
    dj_circ=QuantumCircuit(n+1,n)
    for qubit in range(n):
        dj_circ.h(qubit)
    dj_circ.x(n)
    dj_circ.h(n)
    if case=='random':
        random=np.random.randint(2)
        if random==0:
            case= 'constant'
        if random ==1:
            case='balanced'
    oracle=dj_oracle(case,n)
    dj_circ.append(oracle,range(n+1))
    for qubit in range(n):
        dj_circ.h(qubit)
        dj_circ.measure(qubit,qubit)
    return dj_circ
    
    

```

## Grover Search Algorithm

### Overview
You have likely heard that one of the many advantages a quantum computer has over a classical computer is its superior speed searching databases. Grover's algorithm demonstrates this capability. This algorithm can speed up an unstructured search problem quadratically, but its uses extend beyond that; it can serve as a general trick or subroutine to obtain quadratic run time improvements for a variety of other algorithms. This is called the amplitude amplification trick.

Grover's algorithm consists of three main algorithms steps: state preparation, the oracle, and the diffusion operator. The state preparation is where we create the search space, which is all possible cases the answer could take. In the list example we mentioned above, the search space would be all the items of that list.

The oracle is what marks the correct answer, or answers we are looking for, and the diffusion operator magnifies these answers so they can stand out and be measured at the end of the algorithm.


### Implementation

```python
def phase_oracle(n,marked_indices,name="oracle"):
    qc=QuantumCircuit(n,name=name)
    oracle_matrix=np.identity(2**n)
    for marked_indices in marked_indices:
        oracle_matrix[marked_indices,marked_indices]=-1
    qc.unitary(Operator(oracle_matrix),range(n))
    return qc

def diffuser(n):
    qc=QuantumCircuit(n,name="diff")
    qc.h(range(n))
    qc.append(phase_oracle(n,[0]),range(n))
    qc.h(range(n))
    return qc

def grover(n,marked):
    qc=QuantumCircuit(n,n)
    r=int(np.round(np.pi/(4*np.arcsin(np.sqrt(len(marked)/2**n)))-1/2))
    print(f'{n} qubits,basis state {marked} marked,{r} rounds')
    qc.h(range(n))
    for i in range(r):
        qc.append(phase_oracle(n,marked),range(n))
        qc.append(diffuser(n) ,range(n))
    qc.measure(range(n),range(n))
    return qc
# to finally run the code 
n=4
x=np.random.randint(2**n)
y=np.random.randint(2**n)

marked=[x,y]
qc=grover(n,marked)


qc.draw()

```

## Satisfiability problem using Grover Search

### Overview
Grover search algorithm cab also be used to find solution of some logic problems.
Below is an example using inbuilt grover algorithm from qiskit.


### Implementation

```python
from qiskit.aqua.components.oracles import LogicalExpressionOracle

from qiskit.tools.visualization import plot_histogram

log_expr='((d&a) | (b&c)) & ~(a&b)' 
algorithm=Grover(LogicalExpressionOracle(log_expr))

backend=BasicAer.get_backend('qasm_simulator')

results=algorithm.rum(backend)

plot_histogram(results)


```

## Bernstein Vazirani Algorithm

### Overview
The Bernstein-Vazirani algorithm can be seen as an extension of the Deutsch-Jozsa algorithm . It showed that there can be advantages in using a quantum computer as a computational tool for more complex problems than the Deutsch-Jozsa problem.
Instead of the function being balanced or constant as in the Deutsch-Jozsa problem, now the function is guaranteed to return the bitwise product of the input with some string,s. We are expected to find s.
Classically for n input bit string we need n calls to the funcion.
But using this algorithm we need to call the function only once irrespective of the suze of input bit string.


### Implementation

```python
s='011'
n=3
bv=QuantumCircuit(n+1,n)
bv.x(n)
bv.h(n)
    
for i in range(n):
    bv.h(i)
bv.barrier()
s=s[::-1]
for q in range(n):
    if s[q]=='0':
        bv.i(q)
    else:
        bv.cx(q,n)
bv.barrier()           
for i in range(n):
    bv.h(i)
for i in range(n):
    bv.measure(i, i)
    
bv.draw()


```
## Quantum Fourier Transformation

### Overview
The Fourier transform occurs in many different versions throughout classical computing, in areas ranging from signal processing to data compression to complexity theory. The quantum Fourier transform (QFT) is the quantum implementation of the discrete Fourier transform over the amplitudes of a wavefunction. It is part of many quantum algorithms, most notably Shor's factoring algorithm and quantum phase estimation.

In the most simple form , it is nothing but a change of basis of the state to the fourier basis.


### Implementation

```python
def rotations(circuit, n):
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    rotations(circuit, n)
def swap(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    rotations(circuit, n)
    swap(circuit, n)
    return circuit

qc = QuantumCircuit(4)
qft(qc,4)
qc.draw()
 
 #Sample rum
qc = QuantumCircuit(3)
qc.x(0)#Creating state to be transformed
qc.x(1)
qc.draw()
qft(qc,3)
qc.draw()

#Demonstration of the state
qc.save_statevector()
statevector = sim.run(qc).result().get_statevector()
plot_bloch_multivector(statevector)

```

## Quantum Phase Estimation

### Overview
Quantum phase estimation is one of the most important subroutines in quantum computation. It serves as a central building block for many quantum algorithms. The objective of the algorithm is the following:
Given a unitary operator U, the algorithm estimates theta   Here shi is an eigenvector and e^i theta is the corresponding eigenvalue. Since U is unitary, all of its eigenvalues have a norm of 1.


### Implementation

```python
qpe = QuantumCircuit(4, 3)
qpe.x(3)
qpe.draw()
for qubit in range(3):
    qpe.h(qubit)
qpe.draw()
repetitions = 1
for counting_qubit in range(3):
    for i in range(repetitions):
        qpe.cp(math.pi/4, counting_qubit, 3)
    repetitions *= 2
qpe.draw()
qpe.barrier()
qpe = qpe.compose(QFT(3, inverse=True), [0,1,2])
qpe.barrier()
qpe.measure(0,0)

qpe.draw()


```
## Superdense Coding

### Overview
Superdense coding is a procedure that allows someone to send two classical bits to another party using just a single qubit of communication.

Teleportation
Transmit one
qubit using two
classical bits	

Superdense Coding
Transmit two
classical bits
using one
qubit

The teleportation protocol can be thought of as a flipped version of the superdense coding protocol, in the sense that Alice and Bob merely “swap their equipment.”


### Implementation

```python
def create_bell_pair():
    qc = QuantumCircuit(2)
    qc.h(1)
    qc.cx(1, 0)
    return qc
def encode_message(qc, qubit, msg):
    if msg[1] == "1":
        qc.x(qubit)
    if msg[0] == "1":
        qc.z(qubit)
    return qc
def decode_message(qc):
    qc.cx(1, 0)
    qc.h(1)
    return qc
qc = create_bell_pair()
qc.barrier()
message = '10'
qc = encode_message(qc, 1, message)
qc.barrier()

qc = decode_message(qc)

qc.measure_all()

qc.draw()


```
## Simon's Algorithm

### Overview
We are given an unknown blackbox function f, which is guaranteed to be either one-to-one or two-to-one , where one-to-one and two-to-one functions have the following properties:

one-to-one: maps exactly one unique output for every input.

two-to-one: maps exactly two inputs to every unique output.


### Implementation

```python
b = '110'

n = len(b)
simon_circuit = QuantumCircuit(n*2, n)
simon_circuit.h(range(n))    
simon_circuit.barrier()
simon_circuit = simon_circuit.compose(simon_oracle(b))
simon_circuit.barrier()
simon_circuit.h(range(n))
simon_circuit.measure(range(n), range(n))
simon_circuit.draw()


```

## Shor’s Algorithm for factorization of N=pq

### Overview
Pick a number “a” ,coprime to N.
Find the period r of the function 𝑎^𝑟(𝑚𝑜𝑑 𝑁)  i.e. smallest r such that 𝑎^𝑟≡1 (𝑚𝑜𝑑 𝑁).
If r is even then check 𝑥≡a^(r/2) (mod N) else continue with other a.
If 𝑥+1≢0(𝑚𝑜𝑑 𝑁) then atleast of of {p,q} is contained in {gcd⁡(𝑥+1,𝑁),gcd⁡(𝑥−1,𝑁) }   else start with other a.
So using this algorithm we have converted our question from factorization to period finding .

Using the quantum shor algorithm provides a exponential speedup to the best classical counterparts. It is a crucial algorithm towards cybersecurity as the RSA encryption is based on public key sharing scheme where the private key for decryption is the prime factor of public key.
Suppose our public key is semi prime digit of 226 digits. Then on a classical computer it can take over 1000 years to find its prime factor.



### Implementation

```python
from qiskit import QuantumCircuit


def a_x_mod15(a, x):
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")
    U = QuantumCircuit(4)        
    for iteration in range(x):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, x)
    c_U = U.control()
    return c_U

def modular_exponentiation(given_circuit, n, m, a):
    for x in range(n):
        exponent = 2**x
        given_circuit.append(a_x_mod15(a, exponent), 
                     [x] + list(range(n, n+m)))

#Function for inverse QFT
def apply_iqft(given_circuit, measurement_qubits):

    given_circuit.append(QFT( len(measurement_qubits), do_swaps=False).inverse(), measurement_qubits)


#Function for shor algorithm
def shor_program(n, m, a):
    shor = QuantumCircuit(n+m, n)

    initialize_qubits(shor, n, m)
    shor.barrier()

    modular_exponentiation(shor, n, m, a)
    shor.barrier()

    apply_iqft(shor, range(n))

    shor.measure(range(n), range(n))
    
    return shor
    
n = 4; m = 4; a = 7
mycircuit = shor_program(n, m, a)
mycircuit.draw()
 
###Classical post processing

for measured_value in counts:
    print(f"Measured {int(measured_value[::-1], 2)}")

from math import gcd

for measured_value in counts:
    measured_value_decimal = int(measured_value[::-1], 2)
    print(f"Measured {measured_value_decimal}")
    
    if measured_value_decimal % 2 != 0:
        print("Failed. Measured value is not an even number")
        continue
    x = int((a ** (measured_value_decimal/2)) % 15)
    if (x + 1) % 15 == 0:
        print("Failed. x + 1 = 0 (mod N) where x = a^(r/2) (mod N)")
        continue
    guesses = gcd(x + 1, 15), gcd(x - 1, 15)
    print(guesses)

```

## Quantum Counting

### Overview
Whereas Grover’s algorithm attempts to find a solution to the Oracle, the quantum counting algorithm tells us how many of these solutions there are. This algorithm is interesting as it combines both quantum search and quantum phase estimation.



### Implementation

```python

##Forming a Grover search operator for 5 solutions
def grover_operator(n_iterations):
    from qiskit.circuit.library import Diagonal, GroverOperator
    oracle = Diagonal([1,1,-1,1,1,1,1,-1,1,1,-1,-1,1,1,-1,1])
    grover_it = GroverOperator(oracle).repeat(n_iterations).to_gate()
    grover_it.label = f"Grover$^{n_iterations}$"
    return grover_it

from qiskit.circuit.library import QFT
iqft = QFT(4, inverse=True).to_gate()
iqft.label = "QFT†"

t = 4   
n = 4   
qc = QuantumCircuit(n+t, t)
for qubit in range(t+n):
    qc.h(qubit)
n_iterations = 1
for qubit in range(t):
    cgrit = grover_operator(n_iterations).control()
    qc.append(cgrit, [qubit] + list(range(t, n+t)))
    n_iterations *= 2
    
qc.append(iqft, range(t))
qc.measure(range(t), range(t))
qc.draw(fold=-1)

##Calculation for M that is no of solutions
def calculate_M(measured_int, t, n):
    theta = (measured_int/(2**t))*math.pi*2
    print("Theta = %.5f" % theta)
    N = 2**n
    M = N * (math.sin(theta/2)**2)
    print(f"No. of Solutions = {M:.1f}")


```
