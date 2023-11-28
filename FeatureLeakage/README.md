# Introduction

Idea is to trian a secondary model whos sole purpose is to predicte the sensitive
data from the outputs.

As far as I know there are no guarantees to this (PR this if I am wrong).

Therefore this is more of an optimzation without guarantees.

Basically just have a penalty to make sure the encoder doesnt not encode information
that allows the adversary to retreive it.

# See also:

1. Fair represnetation learning.
