# How Much of a Member's Solo Gain Reaches the Blend?

I retrained one member of a Name That LLM blend on cleaner data. Alone, its public score jumped from 0.522 to 0.566. Back
in the blend, four weightings scored 0.620-0.624 against 0.626 before. Is that surprising?

This notebook simulates it: 18 balanced classes, members with shared and private Gaussian noise, one member replaced by a
better one at fixed weight, or a new member added. It reruns the whole grid (a few minutes on CPU) and compares it with the
saved dataset.
