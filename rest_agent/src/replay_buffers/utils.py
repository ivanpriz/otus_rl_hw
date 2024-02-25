from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
TransitionBatch = namedtuple('TransitionBatch', ('state', 'action', 'next_state', 'reward'))
