"""Check that a NumPy Generator can be checkpointed (pickle or bit_generator.state) and resumed exactly."""
import copy, pickle
import numpy as np
g = np.random.default_rng(np.random.SeedSequence(42, spawn_key=(7,)))
g.standard_normal(1000)
st = copy.deepcopy(g.bit_generator.state)
blob = pickle.dumps(g)
a = g.standard_normal(3)
b = pickle.loads(blob).standard_normal(3)
g3 = np.random.default_rng(); g3.bit_generator.state = st; c = g3.standard_normal(3)
print(f"numpy {np.__version__}: pickled Generator resumes identically: {np.array_equal(a, b)} | "
      f"state-dict restore identical: {np.array_equal(a, c)} | state keys: {list(st.keys())} | pickle bytes: {len(blob)}")
