'''
This script aims to provide a way to assign pre- and post-course genbiomaps
assessment that satisfy the following constraints:

    1. The question items in the two QuestionSet's are disjoint
    2. The number of questions in each QuestionSet is the same
    3. The number of sub-parts in each QuestionSet differ by at most `diffMax`
    4. The number of sub-parts in each QuestionSet that falls within any
        category is greater than or equal to `typeMin`
    5. The ratio of question with answer T (true) falls within the range 
        between `Tmin` and `Tmax` for each QuestionSet
    6. The sub-categories that appear in each QuestionSet matches

While the above constraints are genbiomap oriented it may also be suitable 
to use for similar assessments. (For details of the algorithm, consult the
docstring of the PairOptimizer class).

This script does not require any third-party python packages beyond those
included with the python installer.

This script can either be imported as a python module (e.g., to then be use
inside a jupyter notebook) or run as a command-line script.

In the former case, the main program entry points would be to read a database
of question info using the `read_question_info()` function, and then 
initialize and run an instance of `PairOptimizer` to produce optimal 
pre- and post-course assessment pair. The optimal pair can then examined and
further manipulated by extracting the `best` attribute of the PairOptimizer
instance.

In addition to `PairOptimizer`, this script also provides the following 
classes:
 - QuestionInfo: encapsulate data regarding a single question item
 - QuestionSet: encapsulate a collection of question items
 - PrePostPair: encapsulate a pair of pre-/post- course QuestionSet's

When used in command-line, this script takes the names of the question 
database file and the output file as mandatory arguments, and can optionally
configure various parameters using a .json file. In addition, the script also
takes a verbosity as an additional optional argument.

The parameters that can be set in the .json file are:
 - qNum (int): the number of question in each QuestionSet 
 - typeMin (int): the minimum number of questions in each category
 - diffMax (int): the maximum difference in number of sub-parts
 - TMin (float): the minimum portion of sub-parts with T (true) as answer
 - TMax (float): the maximum portion of sub-parts with T (true) as answer
 - stepMax (int): the maximum number of steps to take in one optimization run
 - badMax (int): the maximum number of "bad" steps in one optimization run
 - badSeqMax (int): the maximum number of consecutive "bad" steps in one 
    optimization run
 - effort (float): the amount of "effort" used in searching through neighbors
     in each step (should be a value between 0 and 1)
 - verbosity (int): the verbosity in the printed log (should be an integer 
    between 0 and 3, inclusive)
 - n_runs (int): the number of independent optimization runs
 - header_row (int): the number of rows used as header in the database csv
 - label_idx (int): the index corresponding to question label in database csv
 - count_idx (int): the index corresponding to sub-part count in database csv
 - TF_idx (int): the index corresponding to T counts in database csv, which 
    should be followed by the column for F counts
 - type_idx (int): the index of the first column of category counts
 - num_type (int): the number of category-count columns
 - subtype_idx (int): the index of the first column of sub-category counts
 - num_subtype (int): the number of sub-category-count columns
 - write_mode ("w" or "a"): the mode under which the output csv is written to

Note that all indexes start from 0, per python's convention

To see the default values used in the above parameters use the -j [JSON] flag
from the command line
'''

import csv, random, copy, math, itertools, dataclasses, collections

#### Class definitions

@dataclasses.dataclass
class QuestionInfo:
    '''
    data class for storing the attributes of a question item

    attributes:
    - label: the label of the question item
    - count: the number of sub-parts
    - Tcount: the number of sub-parts whose answer is T (true)
    - Fcount: the number of sub-parts whose answer is F (false)
    - types: the categories of the sub-parts, as a collection.Counter
    - subtypes: the sub-categories of the sub-parts, as a collection.Counter
    '''

    label: str
    count: int
    Tcount: int
    Fcount: int
    types: collections.Counter[str]
    subtypes: collections.Counter[str]
    
    def __copy__(self):
        '''
        make a copy of the question item data

        NOTE: Both types and subtypes are shallow-copied

        output: a new QuestionInfo instance
        '''
        return self.__class__(
            self.label, self.count, self.Tcount, self.Fcount, 
            self.types.copy(), self.subtypes.copy()
        )
    
    def copy(self):
        '''
        alias of __copy__(self)
        '''
        return self.__copy__()

class QuestionSet:
    '''
    class representing a collection of question items as a question set

    attributes:
    - items: a python set of labels representing the questions selected
    - database: a reference to the database (python dict) that resolves the 
        attributes of the question item with the said labels

    NOTE: the database attributes should be treated a REFERENCE to the
        database rather than a copy. For example, it is expected that 
        multiple QuestionSet would "point to" the same database, and 
        that the database is reference-copied under copy()
    '''
    
    def __init__(self, items, database):
        '''
        initialize a new QuestionSet instance

        arguments:
        - items: an iterable of labels of question items
        - database: a python dict of QuestionInfo, where the key of an item
            should agree with the label attribute of its value
        '''
        self.items = set(_i for _i in items)
        self.database = database
    
    @classmethod
    def sample_from(cls, database, n):
        '''
        create a new QuestionSet instance by random sampling the underlying 
        database

        arguments:
        - database: a python dict of QuestionInfo to be assigned the database 
            attribute and for which the question items are to be sampled from
        - n: the number of question item to be sampled

        returns: a new QuestionSet instance
        '''
        choices = random.sample(list(database.keys()), n)
        return QuestionSet(choices, database)        
    
    def tally_subparts(self):
        '''
        tally the total number of sub-parts in the question set

        returns: the total number of sub-parts
        '''
        return sum(self.database[_i].count for _i in self.items)
    
    def tally_Ts(self):
        '''
        tally the total number of sub-parts with answer T (true) in the 
        question set

        returns: the total number of T's
        '''
        return sum(self.database[_i].Tcount for _i in self.items)
    
    def tally_Fs(self):
        '''
        tally the total number of sub-parts with answer F (false) in the
        question set

        returns: the total number of F's
        '''
        return sum(self.database[_i].Fcount for _i in self.items)
    
    def calc_T_ratio(self):
        '''
        calculate the ratio of sub-parts with answer T (true) in the
        question set

        returns: ratio of T's over the number of sub-parts
        '''
        num_T = self.tally_Ts()
        num_F = self.tally_Fs()
        return num_T / (num_T + num_F)
    
    def tally_types(self):
        '''
        tally the number of sub-parts that falls into particular categories

        returns: a collection.Counter with keys being category labels and 
            values being the corresponding count
        '''
        out_counter = collections.Counter()
        for _i in self.items:
            out_counter.update(self.database[_i].types)
        return out_counter
    
    def tally_subtypes(self):
        '''
        tally the number of sub-parts that falls into particular sub-categories

        returns: a collection.Counter with keys being sub-category labels and
            values being the corresponding count
        '''
        out_counter = collections.Counter()
        for _i in self.items:
            out_counter.update(self.database[_i].subtypes)
        return out_counter
    
    def overlap(self, other, *, sloppy=True):
        '''
        extract items from the QuestionSet that overlaps with those in `other`

        arguments:
        - other: EITHER an iterable of labels of question item (in which case
            these are assumed to have the same database as that of the 
            QuestionSet instance); OR another QuestionSet instance

        keyword-only argument:
        - sloppy: if True, compares only the labels for overlap without 
            checking if the questions are indeed equivalent in the 
            underlying database (note: this parameter is used only when `other` 
            is another QuestionSet instance)

        returns: A python set of labels where overlap occurs
        '''
        if isinstance(other, QuestionSet):
            out = self.items.intersection(other.items)
            if sloppy:
                return out
            else:
                return {
                    _k for _k in out if 
                    self.database[_k]==other.database[_k]
                }
        else:
            return self.items.intersection(other)

    def __str__(self):
        '''
        provide a simplified string representation of the QuestionSet instance

        NOTE: database is represented by its python object ID
        '''
        return str(self.items) + "; database@" + str(id(self.database))
    
    def __repr__(self):
        '''
        provide a raw string representation of the QuestionSet instance

        NOTE: database is represented by its python object ID
        '''
        return repr(self.items) + "; database@" + repr(id(self.database))
    
    def __copy__(self):
        '''
        create a copy of the QuestionSet

        returns: a new QuestionSet instance

        NOTE: the items attribute is essentially shallow-copied, while
        the database attribute is reference-copied
        '''
        return self.__class__(self.items, self.database)
    
    def copy(self):
        '''
        alias of __copy__(self)
        '''
        return self.__copy__()
    
    def export(self, sort=True):
        '''
        export the items in the QuestionSet as a list of labels

        arguments:
        - sort: if True, the list is sorted by using the built-in key

        returns: a python list of the labels of questions
        '''
        if sort:
            return sorted(self.items)
        else:
            return list(self.items)

class PrePostPair:
    '''
    class representing a pair of pre- and post-course question sets

    attributes:
    - pre: the pre-course question set, as a QuestionSet instance
    - post: the post-course question set, as a QuestionSet instance
    '''
    
    def __init__(self, pre, post, database):
        '''
        initialize a new PrePostPair instance

        arguments:
        - pre: an iterable of labels of pre-course question items
        - post: an iterable of labels of post-course question items
        - database: EITHER a python dict of QuestionInfo, which would be use 
            as common database for both pre- and post-course items; OR a 
            2-tuple of python dict of QuestionInfo, for which the 0-th dict
            would be used as database of pre, and the 1-st dict would be used
            as database of post
        '''
        try:
            self.pre = QuestionSet(pre, database[0])
            self.post = QuestionSet(post, database[1])
        
        except KeyError:
            self.pre = QuestionSet(pre, database)
            self.post = QuestionSet(post, database)
            
    @classmethod
    def from_qsets(cls, pre, post, *, copy=False):
        '''
        create a new PrePostPair instance from a pair of existing QuestionSet
        instances

        arguments:
        - pre: an iterable of labels of pre-course question items
        - post: an iterable of labels of post-course question items
        - database: EITHER a python dict of QuestionInfo, which would be use 
            as common database for both pre- and post-course items; OR a 
            2-tuple of python dict of QuestionInfo, for which the 0-th dict
            would be used as database of pre, and the 1-st dict would be used
            as database of post

        keyword-only arguments:
        - copy: if True, the PrePostPair will contains COPIES of the original
            QuestionSet instance (i.e., same content, different container);
            Otherwise, the original QuestionSet instances are directly used
            in constructing the PrePostPair (i.e., same reference)
        
        returns: a new PrePostPair instance
        '''
        obj = cls.__new__(cls)
        super(cls, obj).__init__()
        
        if copy:
            obj.pre = pre.copy()
            obj.post = post.copy()
        else:
            obj.pre = pre
            obj.post = post
            
        return obj
    
    @classmethod
    def sample_from(cls, database, n, n2=None, *, disjoint=False):
        '''
        create a new PrePostPair instance by taking a random sample from a
        given database

        arguments:
        - database: the database to be used as database of both the pre and 
            post QuestionSet instances, and for which the question items are
            sampled from
        - n: if n2 is None, the number of question items contained in each
            of the pre- and post- QuestionSet instances; if n2 is not None, 
            the number of question items contained in the pre QuestionSet
        - n2: if not None, the number of question items contained in the
            post QuestionSet instance

        keyword-only arguments:
        - disjoint: if True, question items in the pre QuestionSet will be 
            disjoint (no overlap) from the items in the post QuestionSet
        
        returns: a new PrePostPair instance
        '''
        n2 = n if (n2 is None) else n
        keys = list(database.keys())
        if disjoint:
            choices = random.sample(keys, n + n2)
            choices1 = choices[:n]
            choices2 = choices[n:]
        else:
            choices1 = random.sample(keys, n)
            choices2 = random.sample(keys, n2)
            
        return cls(choices1, choices2, database)
    
    def subpart_diff(self):
        '''
        Count the difference in the number of sub-parts between the post
        QuestionSet and the pre QuestionSet

        returns: difference (= post - pre) in sub-parts between the pre and 
            post QuestionSets
        '''
        return self.post.tally_subparts() - self.pre.tally_subparts()
    
    def type_diff(self):
        '''
        Count the difference in the number of sub-parts within each category
            between the post QuestionSet and the pre QuestionSet

        returns: python dict with keys being categories and values being 
            the corresponding difference (= post - pre) in number of sub-parts
        '''
        pre = self.pre.tally_types()
        post = self.post.tally_types()
        keys = set(pre.keys()).union(post.keys())
        return {_k: post.get(_k, 0) - pre.get(_k, 0) for _k in keys}

    def type_missed(self):
        '''
        Find the categories present in one QuestionSet but not the other

        returns: 2-tuple of python set. The 0-th (1-st) set consists of 
            categories in the post (pre) QuestionSet but not in the pre 
            (post) QuestionSet
        '''
        pre = {_k for _k, _v in self.pre.tally_types().items() if _v > 0}
        post = {_k for _k, _v in self.post.tally_types().items() if _v > 0}
        return post - pre, pre - post
    
    def subtype_diff(self):
        '''
        Count the difference in the number of sub-parts within each sub-category
            between the post QuestionSet and the pre QuestionSet

        returns: python dict with keys being sub-categories and values being 
            the corresponding difference (= post - pre) in number of sub-parts
        '''
        pre = self.pre.tally_subtypes()
        post = self.post.tally_subtypes()
        keys = set(pre.keys()).union(post.keys())
        return {_k: post.get(_k, 0) - pre.get(_k, 0) for _k in keys}

    def subtype_missed(self):
        '''
        Find the sub-categories present in one QuestionSet but not the other

        returns: 2-tuple of python set. The 0-th (1-st) set consists of 
            sub-categories in the post (pre) QuestionSet but not in the pre 
            (post) QuestionSet
        '''
        pre = {_k for _k, _v in self.pre.tally_subtypes().items() if _v > 0}
        post = {_k for _k, _v in self.post.tally_subtypes().items() if _v > 0}
        return post - pre, pre - post
    
    def tally_subparts(self):
        '''
        tally the total number of sub-parts within each QuestionSet

        returns: the number of sub-parts in each QuestionSet, as a (pre, post)
            2-tuple
        '''
        return self.pre.tally_subparts(), self.post.tally_subparts()
    
    def tally_Ts(self):
        '''
        tally the total number of sub-parts with answer T (true) within each
        QuestionSet

        returns: the number of T's in each QuestionSet, as a (pre, post)
            2-tuple
        '''
        return self.pre.tally_Ts(), self.post.tally_Ts()
    
    def tally_Fs(self):
        '''
        tally the total number of sub-parts with answer F (false) within each
        QuestionSet

        returns: the number of F's in each QuestionSet, as a (pre, post)
            2-tuple
        '''
        return self.pre.tally_Fs(), self.post.tally_Fs()
    
    def calc_T_ratio(self):
        '''
        calculate the ratio of sub-parts with answer T (true) within each
        QuestionSet

        returns: ratio of T's over the number of sub-parts for each 
        QuestionSet, as a (pre, post) 2-tuple
        '''
        return self.pre.calc_T_ratio(), self.post.calc_T_ratio()
        
    def tally_types(self):
        '''
        tally the number of sub-parts that falls into particular categories
        within each QuestionSet

        returns: a (pre, post) 2-tuple of collection.Counter with keys being 
            category labels and values being the corresponding count
        '''
        return self.pre.tally_types(), self.post.tally_types()
    
    def tally_subtypes(self):
        '''
        tally the number of sub-parts that falls into particular sub-categories
        within each QuestionSet

        returns: a (pre, post) 2-tuple of collection.Counter with keys being 
            sub-category labels and values being the corresponding count
        '''
        return self.pre.tally_subtypes(), self.post.tally_subtypes()
    
    def overlap(self, *, sloppy=True):
        '''
        extract items that overlap between the pre- and post- QuestionSet's

        keyword-only argument:
        - sloppy: if True, compares only the labels for overlap without 
            checking if the questions are indeed equivalent in the 
            underlying database

        returns: A python set of labels where overlap occurs
        '''
        return self.pre.overlap(self.post, sloppy=sloppy)

    def __str__(self):
        '''
        provide a simplified string representation of the PrePostPair instance

        NOTE: database is represented by its python object ID
        '''
        return "pre: " + str(self.pre) + "\npost: " + str(self.post)
    
    def __repr__(self):
        '''
        provide a raw string representation of the PrePostPair instance

        NOTE: database is represented by its python object ID
        '''
        return "pre: " + repr(self.pre) + "\npost: " + repr(self.post)
    
    def __copy__(self):
        '''
        create a copy of the PrePostPair

        returns: a new PrePostPair instance

        NOTE: the items attribute in each QuestionSet is essentially 
            shallow-copied, while the corresponding database attribute is 
            reference-copied
        '''
        return self.__class__.from_qsets(self.pre, self.post, copy=True)
    
    def copy(self):
        '''
        alias of __copy__(self)
        '''
        return self.__copy__()
    
    def export(self, sort=True, label=True):
        '''
        export the items in the PrePostPair

        arguments:
        - sort: if True, the list is sorted by using the built-in key
        - label: if True, return a dictionary with keys ["pre", "post"], 
            otherwise return a (pre, post) 2-tuple

        returns: if label==True, a python dict with keys ["pre", "post"], 
            and values the corresponding list of question items; otherwise
            a (pre, post) 2-tuple of list of the corresponding question items
        '''
        if label:
            return {
                "pre": self.pre.export(sort=sort), 
                "post": self.post.export(sort=sort)
            }
        else:
            return (self.pre.export(sort=sort), self.post.export(sort=sort))

    def is_database_equiv(self):
        '''
        Check if the database for pre and post QuestionSet are equivalent

        return: boolean of equivalence
        '''
        return (self.pre.database == self.post.database)

    def get_database(self):
        '''
        get the database(s) of the pre and post QuestionSet

        returns: if the two QuestionSets are equivalent, a single database
            (i.e., dict of QuestionInfo); otherwise, a (pre, post) 2-tuple
            of databases

        NOTE: database(s) is(are) returned by reference
        '''
        db1 = self.pre.database
        db2 = self.post.database
        if db1 == db2:
            return db1
        else:
            return (db1, db2)
    
    def get_items(self, *, copy=True):
        '''
        get the labels of question items of the pre and post QuestionSet's

        keyword-only arguments:
        - copy (defualt=True): if True, the question items are shallow-copied,
            otherwise they are returned by reference

        returns: a (pre, post) 2-tuple of sets
        '''
        if copy:
            return (self.pre.items.copy(), self.post.items.copy())
        else:
            return (self.pre.items, self.post.items)
        
    database = property(get_database)
    items = property(get_items)

class PairOptimizer:
    '''
    class encapsulating the optimization process of a PrePostPair

    The optimizer aims to create pre-post pairs with the following constraints:
    1. The question items in the two QuestionSet's are disjoint
    2. The number of questions in each QuestionSet is the same
    3. The number of sub-parts in each QuestionSet differ by at most `diffMax`
    4. The number of sub-parts in each QuestionSet that falls within any
        category is greater than or equal to `typeMin`
    5. The ratio of question with answer T (true) falls within the range 
        between `Tmin` and `Tmax` for each QuestionSet
    6. The sub-categories that appear in each QuestionSet matches

    Constraints (1) and (2) are always satisfied by construction; the optimizer 
    attempts to reach the remaining remaining constraints through a two-tier 
    local search algorithm with possible reset, wherein constraints (3) - (5)
    are to be satisfied before (6) is attempted

    For each step in the local search, the algorithm considers any alternative
    PrePostPair that results from one swap of a used question with an unused 
    question a potential candidate.

    The local search algorithm will keep searching until an optimal state is 
    reached, or when one of the following termination condition is reached:
    a. the number of steps taken is bigger than `stepMax`
    b. the number of "bad" (non-improving) steps taken is bigger than `badMax`
    c. the number of consecutive bad steps taken is bigger than `badSeqMax`

    Note that unless the termination condition is reached, the algorithm will
    always take a step, even when there are no improving PrePostPair among the
    candidates it considers. Instead, the algorithm retains a "best" candidate
    that it has obtained so far in the entire search process.

    public attributes:
    - database: the database (python dict of QuestionInfo with matching keys)
        for which question items can be pooled from
    - qNum: the number of questions in each pre- and post- QuestionSet
    - typeMin: the minimum number of sub-parts required for each QuestionSet
        and in each category
    - diffMax: the maximum difference in the number of sub-parts between the
        pre- and post- QuestionSet's
    - TMin: the minimum ratio of sub-parts with T (true) being the answer for
        each QuestionSet
    - Tmax: the maximum ratio of sub-parts with T (false) being the answer for
        each QuestionSet
    - badSeqMax: the maximum number of consecutive bad steps the algorithm may 
        take before terminating
    - badMax: the maximum number of bad steps the algorithm may take before
        terminating
    - stepMax: the maximum number of steps the algorithm may take before 
        terminating
    - effort: parametrize the proportion of nearest neighbors searched in each
        step. Should be a float between 0 and 1 (note: the number
        of nearest neighbors searched actually scales as effort**2)
    - verbosity: the verbosity of logging. Should be an integer between 0 
        (least verbose) to 3 (most verbose)
    - logger: the function used to log messages. Should have the same interface
        as the built-in print function, in particular, it should accept the 
        `end` and `sep` keyword arguments
    - best: a PrePostPair instance representing the best candidate for 
        satisfying all constraints (or None if no candidate has been 
        identified)
    - current: a PrePostPair instance representing the current state in the 
        local search process
    - step: an integer counter on the number of steps taken
    - bad: an integer counter on the number of "bad" steps taken
    - badSeq: an integer counter on the number of consecutive "bad" steps taken

    private attributes:
    - _summary: cache for storing summary information of QuestionSets, used
        to construct score for constraints (3) - (5) violations
    - _cnstrScale: scale factors used to construct constraint violation scores
        from constraints (3) - (5) violations, in the respective order
    - _cnstrScore: cache of constraint score for constraints (3) - (5) 
        violations for the current state
    - _optScore: cache of constraint score for constraint (6) violation for the
        current state
    - _bestCnstrSc: cache for constraint score for constraints (3) - (5) 
        violations for the best candidate (= math.inf if no candidate has been 
        identified)
    - _bestOptSc: cache for constraint score for constraint (6) violation for
        the best candidate (= math.inf if no candidate has been identified)
    - _backtrack: cache of the previous step taken, in the form of (old_label, 
        new_label, loc), where loc = 0 (= 1) corresponds to the pre (post) 
        QuestionSet. Used to prevent taking a immediate reversal step during 
        local search
    '''

    def __init__(self, database, qNum, typeMin, diffMax, TMin, TMax, stepMax,
        badMax, badSeqMax, effort = 0.5, verbosity = 1, logger = print
    ):
        '''
        initialize a new PairOptimizer instance

        argument names generally agrees to the corresponding attribute name
        '''
        self.database = database
        self.qNum = qNum
        self.typeMin = typeMin
        self.diffMax = diffMax
        self.TMin = TMin
        self.TMax = TMax
        self.badSeqMax = badSeqMax
        self.badMax = badMax
        self.stepMax = stepMax
        self.effort = effort
        self.verbosity = verbosity
        self.logger = logger
        self.best = None
        self._bestCnstrSc = math.inf
        self._bestOptSc = math.inf
        self._backtrack = None
        self.current = None
        self.step = 0
        self.badSeq = 0
        self.bad = 0
        self._summary = None
        self._cnstrScale = (1, 1, 10)
        self._cnstrScore = None
        self._optScore = None
    
    def reset_counters(self):
        '''
        reset the step counters (step, bad, badSeq) of the optimizer instance
        '''
        self.step = 0
        self.badSeq = 0
        self.bad = 0
    
    def incr_counters(self, bad):
        '''
        reset the step counters (step, bad, badSeq) of the optimizer instance

        argument:
        - bad: if true, increment assuming a "bad" step has been taken. 
            Otherwise, increment assuming a good step has been taken.
        '''
        self.step += 1
        if bad:
            self.bad += 1
            self.badSeq += 1
        else:
            self.badSeq = 0

    def _initiate(self):
        '''
        Initiate the local search process by populating the cache _summary,
        _cnstrScore, and _optScore
        '''
        self._summarize_cnstr()
        self._score_cnstr()
        self._score_opt()
        if self.verbosity > 1:
            self.logger("Step #{}:".format(self.step), end=" ")
            self.logger(
                "Constraint Score={:.2f},".format(self._cnstrScore), end=" "
            )
            self.logger("Optimization Score={}".format(self._optScore))
    
    def update_best(self):
        '''
        Check if the current state is more optimal than the stored best state, 
        and update the best state should that be the case

        NOTE: update will NOT occur in case of tie, and _optScore is 
        tie-breaking only if _cnstrScore==0
        '''
        changed = False

        # case 1: only _cnstrScore matters
        if (self._cnstrScore < self._bestCnstrSc):
            self.best = self.current.copy()
            self._bestCnstrSc = self._cnstrScore
            self._bestOptSc = self._optScore
            changed = True
        
        # case 2: _cnstrScore == 0, only _optScore matters
        elif (not self._cnstrScore) and (self._optScore < self._bestOptSc):
            self.best = self.current.copy()
            self._bestCnstrSc = self._cnstrScore
            self._bestOptSc = self._optScore
            changed = True
            
        if changed and self.verbosity > 1:
            self.logger("Best pre-post pair updated.")
            if self.verbosity > 2:
                self.logger(str(self.best))

    def reset_best(self):
        '''
        reset the current best candidate to None; and reset the corresponding
        cached scores _bestCnstrSc and _bestOptSc
        '''
        self.best = None
        self._bestCnstrSc = math.inf
        self._bestOptSc = math.inf         
        
    def export_best(self, sort=True, label=True):
        '''
        export the best PrePostPair candidate using its `export` method

        for meaning of arguments and signature of returns, see the 
        documentation of the `export` method for PrePostPair
        '''
        if self.best is None:
            raise ValueError("Best pre-post pair does not exist")
        else:
            return self.best.export(sort=sort, label=label)

    def export_current(self, sort=True, label=True):
        '''
        export the current PrePostPair state using its `export` method 

        for meaning of arguments and signature of returns, see the 
        documentation of the `export` method for PrePostPair
        '''
        if self.current is None:
            raise ValueError("current pre-post pair does not exist")
        else:
            return self.current.export(sort=sort, label=label)

    def iter_prepost(self, effort, *, seed=None):
        '''
        return an iterator that iterates through a selection of question 
        items in the current PrePostPair state, with each next() yielding a 
        2-tuple (loc, label), where loc = 0 (= 1) for pre (post) QuestionSet,
        and label is the label of question item

        argument:
        - effort: the portion of question items to yield

        keyword-only argument:
        - seed: the seed for randomizing the select and order for yield

        NOTE 1: the yield of this generator always alternative between the 
        pre and the post QuestionSet, but may start with either

        NOTE 2: each time this generator is called, a new selection and order 
        for yielding the question items is used. To get deterministic 
        behavior use the seed argument
        '''
        if seed is not None:
            random.seed(seed)
    
        parity = random.randrange(0, 2)
    
        pre = list(self.current.pre.items)
        random.shuffle(pre)
        
        post = list(self.current.post.items)
        random.shuffle(post)
        
        # the trick of using itertools.chain.from_iterable(zip(...)) to
        # interleave multiple iterators comes from more-itertools under
        # interleave()

        if parity:
            parity_iter = itertools.cycle([1, 0])
            item_iter = itertools.chain.from_iterable(zip(post, pre))
        else:
            parity_iter = itertools.cycle([0, 1])
            item_iter = itertools.chain.from_iterable(zip(pre, post))
            
        n = math.ceil( (len(pre) + len(post)) * max(0, min(effort, 1)) )
        return itertools.islice(zip(parity_iter, item_iter), n)

    def list_prepost(self, effort, *, seed=None):
        '''
        return a list of a selection of question items in the current 
        PrePostPair state

        essentially a list() wrapper around iter_prepost(). See the 
        iter_prepost() documentation for meaning of arguments and signature
        of each element in the returned list
        '''
        return list(self.iter_prepost(effort, seed=seed))
    
    def iter_unused(self, effort, *, seed=None):
        '''
        return an iterator that iterates through a selection of question 
        items not used in the current PrePostPair state, with each next() 
        yielding a question item label

        argument:
        - effort: the portion of question items to yield

        keyword-only argument:
        - seed: the seed for randomizing the select and order for yield

        NOTE: each time this generator is called, a new selection and order 
        for yielding the question items is used. To get deterministic 
        behavior use the seed argument
        '''
        if seed is not None:
            random.seed(seed)
        
        avail = set(self.database.keys()) - self.current.pre.items - self.current.post.items
        avail = list(avail)
        random.shuffle(avail)
        n = math.ceil( len(avail) * max(0, min(1, effort)) )
        return itertools.islice(avail, n)
    
    def list_unused(self, effort, *, seed=None):
        '''
        return a list of a selection of question items not used in the 
        current PrePostPair state

        essentially a list() wrapper around iter_unused(). See the 
        iter_unused() documentation for meaning of arguments and signature
        of each element in the returned list
        '''
        return list(self.iter_unused(effort, seed=seed))
    
    def set_params(self, **kwargs):
        '''
        set the attributes of the optimizer relevant to the optimization,
        each attribute is supplied via keyword argument(s) whose name matches
        that used in __init__()

        attributes that can be set:
            qNum, typeMin, diffMax, TMin, TMax, badSeqMax, badMax, stepMax,
            effort, verbosity, logger
        (other keyword arguments are simply ignored)
        '''
        self.qNum = kwargs.get("qNum", self.qNum)
        self.typeMin = kwargs.get("typeMin", self.typeMin)
        self.diffMax = kwargs.get("diffMax", self.diffMax)
        self.TMin = kwargs.get("TMin", self.TMin)
        self.TMax = kwargs.get("TMax", self.TMax)
        self.badSeqMax = kwargs.get("badSeqMax", self.badSeqMax)
        self.badMax = kwargs.get("badMax", self.badMax)
        self.stepMax = kwargs.get("stepMax", self.stepMax)
        self.effort = kwargs.get("effort", self.effort)
        self.verbosity = kwargs.get("verbosity", self.verbosity)
        self.logger = kwargs.get("logger", self.logger)
    
    def sample_pair(self):
        '''
        initialize the current state for local search by randomly sampling 
        from the underlying database
        '''
        self.current = PrePostPair.sample_from(
            self.database, self.qNum, disjoint=True
        )   
        if self.verbosity > 2:
            self.logger(str(self.current))
        
    def set_pair(self, pre, post):
        '''
        initialize the current state for local search by specifying the 
        labels of question for the QuestionSet's

        arguments:
        - pre: labels of question items in the pre QuestionSet
        - post: labels of question items in the post QuestionSet
        '''
        self.current = PrePostPair(pre, post, self.database)
            
    def _summarize_cnstr(self):
        '''
        construct the summary object relevant for construction of constraint 
        score on constraints (3) - (5) violation from the current local 
        search state, and store the result in the _summary attribute
        '''
        pair = self.current
        self._summary = [
            list(pair.tally_subparts()),
            list(pair.tally_Ts()),
            list(pair.tally_types())
        ]
    
    def _score_cnstr(self, *, recalc_all = False):
        '''
        compute the constraint violation score corresponding to constraints
        (3) - (5) for the current local search state, and store the result in 
        the _cnstrScore attribute

        argument:
        - recalc_all: if True, the _summary attribute is recomputed

        NOTE: rely on accuracy of the cached _summary attribute
        '''
        
        if recalc_all or (self._summary is None):
            self._summarize_cnstr()

        # contribution from bounds on T ratios
        # max(self.TMin - _r, _r - self.TMax, 0) results in the "funnel" function 
        # shape ( \_/ ) with kinks at `TMin`` and `TMax`
        tmp = [ self._summary[1][_i] / self._summary[0][_i] for _i in (0, 1) ]
        score = sum(max(self.TMin - _r, _r - self.TMax, 0) for _r in tmp)
        score *= self._cnstrScale[2]
        
        # contribution from difference in number of sub-parts
        tmp = max(0, abs(self._summary[0][0] - self._summary[0][1]) - self.diffMax)
        score += self._cnstrScale[0] * tmp
        
        # contribution from category deficiency
        score += self._cnstrScale[1] * sum( 
            sum(
                max(self.typeMin - _v, 0) 
                for _v in self._summary[2][_i].values()
            ) for _i in (0, 1)
        )

        self._cnstrScore = score
    
    def _score_swap_cnstr(self, old, new, loc, *, recalc_all = False):
        '''
        compute the constraint violation score corresponding to constraints
        (3) - (5), for the state in which the current state has question item
        labeled `old` replaced by an unused item `new`, where the old item
        is found in the pre (post) QuestionSet if `loc` = 0 (= 1)

        arguments:
        - old: label of question item to be replaced
        - new: label ot an unused question item to be placed in
        - loc: if 0 (1), the replacement occurs in the pre (post) QuestionSet

        keyword-only argument:
        - recalc_all: if True, the original summary object and constraint 
            scores are recomputed

        returns: a 2-tuple, with 0-th index the new constraint violation 
            score, and 1-st index being the updated summary object

        NOTE 1: rely on accuracy of the the cached _summary

        NOTE 2: temporarily modify _summary and _cnstrScore in the process of
        computing the score for the swapped QuestionSet's

        NOTE 3: there is no check that the unused item is indeed used by the
        current QuestionSet's
        '''
        
        if recalc_all or (self._summary is None):
            self._summarize_cnstr()
        if recalc_all or (self._cnstrScore is None):
            self._score_cnstr()

        # copy old cached values so that they can be temporarily modified
        old_summary = copy.deepcopy(self._summary)
        old_score = self._cnstrScore
        
        # resolve label to QuestionInfo object
        old = self.database[old]
        new = self.database[new]
        
        # update the _summary to the swapped case in-place
        self._summary[0][loc] += new.count - old.count
        self._summary[1][loc] += new.Tcount - old.Tcount
        self._summary[2][loc] += new.types
        self._summary[2][loc] -= new.types

        # use _score_cnstr() to compute new score
        self._score_cnstr()
        
        # extract new summary and score
        summary = self._summary
        score = self._cnstrScore
        
        # restore old summary and score
        self._summary = old_summary
        self._cnstrScore = old_score
        
        return (score, summary)
    
    def _score_opt(self):
        '''
        compute the constraint violation score corresponding to constraint 
        (6) for the current local search state, and store the result in the 
        _optScore attribute
        '''
        pre_missed, post_missed = self.current.subtype_missed()
        diff = self.current.subtype_diff()
        
        score = sum(_v for _k, _v in diff.items() if _k in pre_missed)
        # double negative => positive contribution
        score -= sum(_v for _k, _v in diff.items() if _k in post_missed)
        
        self._optScore = score
    
    def _score_swap_opt(self, old, new, loc):
        '''
        compute the constraint violation score corresponding to constraint
        (6), for the state in which the current state has question item
        labeled `old` replaced by an unused item `new`, where the old item is 
        found in the pre (post) QuestionSet if `loc` = 0 (= 1)

        arguments:
        - old: label of question item to be replaced
        - new: label ot an unused question item to be placed in
        - loc: if 0 (1), the replacement occurs in the pre (post) QuestionSet

        returns: the new constraint violation score

        NOTE: there is no check that the unused item is indeed used by the
        current QuestionSet's
        '''

        # resolve label to QuestionInfo object
        old = self.database[old]
        new = self.database[new]
        
        pre = self.current.pre.tally_subtypes()
        post = self.current.post.tally_subtypes()
        
        if loc: # loc = 1 => post
            post += new.subtypes
            post -= old.subtypes
        else: # loc = 0 => pre
            pre += new.subtypes
            pre -= old.subtypes
        
        # compute the new subtype diff's
        keys = set(pre.keys()).union(post.keys())
        diff = {_k: post.get(_k, 0) - pre.get(_k, 0) for _k in keys}
        
        # compute the new subtype missed
        pre = {_k for _k, _v in pre.items() if _v > 0}
        post = {_k for _k, _v in post.items() if _v > 0}
        pre_missed, post_missed = post - pre, pre - post

        score = sum(_v for _k, _v in diff.items() if _k in pre_missed)
        # double negative => positive contribution
        score -= sum(_v for _k, _v in diff.items() if _k in post_missed)
        
        return score        

    def move(self):
        '''
        take a single step in the local search algorithm, updating current,
        _summary, _cnstrScore, _optScore, _backtrack, step, bad, and badSeq
        in the process
        '''
        # these are best NEIGHBORS, not global best
        best = None
        best_smmry = None
        best_cnstr_sc = math.inf
        best_opt_sc = math.inf
        
        # reuse the same shuffling selection/order for unused items
        avail = self.list_unused(self.effort)

        for _loc, _old in self.iter_prepost(self.effort):
            for _new in avail:
                
                # skip if results in immediate backtrack
                if (_new, _old, _loc) == self._backtrack:
                    continue
                
                # first check if cnstrScore is best (smallest) so far
                cnstr_sc, smmry = self._score_swap_cnstr(_old, _new, _loc)
                if (cnstr_sc < best_cnstr_sc):
                    best = (_old, _new, _loc)
                    best_cnstr_sc = cnstr_sc
                    best_smmry = smmry
                    if not cnstr_sc: # equiv to cnstr_sc == 0
                        best_opt_sc = self._score_swap_opt(_old, _new, _loc)

                # if there's a state with cnstrScore == 0, check if optScore
                # is best (smallest) so far
                elif not cnstr_sc: # equiv to cnstr_sc == 0
                    opt_sc = self._score_swap_opt(_old, _new, _loc)
                    if (opt_sc < best_opt_sc):
                        best = (_old, _new, _loc)
                        best_cnstr_sc = cnstr_sc
                        best_smmry = smmry
                        best_opt_sc = opt_sc
        
        if best_cnstr_sc: # equiv to best_cnstr_sc > 0
            if (best_cnstr_sc < self._cnstrScore):
                self.incr_counters(False) # improved => "good" step
            else:
                self.incr_counters(True) # "bad" step
        else: # equiv to cnstr_sc == 0
            if (best_opt_sc < self._optScore):
                self.incr_counters(False) # improved => "good" step
            else:
                self.incr_counters(True) # "bad" step
        
        #print(self.step, self.bad, self.badSeq) #DEBUG
        
        self._backtrack = best
        self._cnstrScore = best_cnstr_sc
        self._optScore = best_opt_sc
        self._summary = best_smmry

        # trick: use the fact that _items are REFERENCES to the internal
        # `item` attributes for the QuestionSet's
        _items = self.current.get_items(copy=False)
        _items[best[2]].remove(best[0])
        _items[best[2]].add(best[1])
            
        if self.verbosity > 1:
            self.logger("Step #{}:".format(self.step), end=" ")
            self.logger("Swap = {};".format(best), end=" ")
            self.logger(
                "Constraint Score={:.2f},".format(best_cnstr_sc), end=" "
            )
            self.logger("Optimization Score={}".format(best_opt_sc))
    
    def run(self, reset_start = True):
        '''
        perform a full run of the local search algorithm, populate the `best`
        attribute with the most optimal state obtained

        argument:
        - reset_start: if True, the existing current local search state is
            discarded and replace by a new random sample, and the counters
            step, bad, and badSeq are reset to 0
        '''
        if reset_start:
            self.reset_counters()
            self.sample_pair()
        
        self._initiate()
        self.update_best()

        # success => best state already found
        if (not self._bestCnstrSc) and (not self._bestOptSc):
            success = True
        else:
            success = False
        
        if (not success) and self.verbosity:
            self.logger("Optimization initiated.")
        
        # main loop
        while (
            (not success) and 
            (self.step < self.stepMax) and 
            (self.bad < self.badMax) and 
            (self.badSeq < self.badSeqMax)
        ):
            self.move()
            self.update_best()
            if self.verbosity==1: self.logger(".", end="")
            if (not self._cnstrScore) and (not self._optScore):
                success = True
        
        if self.verbosity:
            if self.verbosity==1: self.logger("")
            if success:
                self.logger("Best outcome obtained. Terminate")
            elif (self.step >= self.stepMax):
                self.logger("Maximum number of steps reached. Terminate")
            elif (self.step >= self.badMax):
                self.logger("Maximum number of bad steps reached. Terminate")
            elif (self.step >= self.badSeqMax):
                self.logger(
                    "Maximum number of consecutive bad steps reached. " + 
                    "Terminate"
                )
            else:
                self.logger("Unexpected termination")
            
            if self.verbosity > 2:
                self.logger(str(self.best))
    
    def run_multiple(self, n):
        '''
        perform multiple full runs of the local search algorithm, populate the
        `best` attribute with the most optimal state obtained. Terminate when
        the maximum number of run is reached or when an optimal state is 
        reached

        argument:
        - n: the maximum number of runs to try
        '''
        for _i in range(n):

            if self.verbosity:
                self.logger("==== RUN #{} ====".format(_i))
            
            self.run(reset_start = True)
            
            # break if best state already reach
            if (not self._bestCnstrSc) and (not self._bestOptSc):
                break

def read_question_info(
    filename, header_row, label_idx, count_idx, TF_idx, type_idx, num_type, 
    subtype_idx, num_subtype, *, base_db = None
):
    '''
    read question items information from csv and process it into a question
    database (i.e., dict of QuestionInfo with matching key and QuestionInfo 
    label)

    arguments (note: all COLUMNS are zero indexed):
    - filename: the name (path) of the csv file
    - header_row: the number of rows that are used as header (note: the last
        header row is expected to contain labels for categories and sub-
        categories)
    - label_idx: the column that contains question item labels
    - count_idx: the column that contains sub-part counts
    - TF_idx: the column that contains sub-part-with-T-as-answer counts, which
        should be followed by the column that contains sub-parts-with-F-as-
        answer counts
    - type_idx: the first column that contains category counts
    - num_type: the number of columns that contain category counts
    - subtype_idx: the first column that contains sub-category counts
    - num_subtype: the number of columns that contain sub-category counts

    keyword-only argument:
    - base_db: if not None, the database (i.e., dict of QuestionInfo ...) for
        which new entries are appended to (note: the base_db is shallow-copied)
    
    returns: a python dict of QuestionInfo
    '''
    if base_db is None:
        out_dict = dict()
    else:
        out_dict = base_db.copy()
        
    with open(filename) as infile:
        csv_reader = csv.reader(infile)
        for _i in range(header_row):
            header = next(csv_reader)
        
        type_names = header[type_idx:type_idx + num_type]
        subtype_names = header[subtype_idx:subtype_idx + num_subtype]
        
        for _row in csv_reader:
            
            label = _row[label_idx]
            count = int(_row[count_idx])
            Tcount = int(_row[TF_idx])
            Fcount = int(_row[TF_idx + 1])
            
            # note: rely on the "stop at first exhaustion" behavior of zip
            types = {
                _k: int(_v) for _k, _v in zip(type_names, _row[type_idx:])
            }
            subtypes = {
                _k: int(_v) for _k, _v in 
                zip(subtype_names, _row[subtype_idx:])
            }
            
            out_dict[label] = QuestionInfo(
                label, count, Tcount, Fcount, types, subtypes
            )
    
    return out_dict

def write_csv(filename, *rows, write_mode="w"):
    '''
    general-purpose function to write python iterable(s) to external csv file

    arguments:
    - filename: the name (path) of the csv file
    - *row: each remaining positional argument is an iterable, and represent 
        a row to be written to the csv file

    keyword-only arguments:
    - write_mode: the mode for which the output file is opened with. Common
        choices are "w" and "a"

    side effect: csv file written
    '''

    with open(filename, write_mode, newline='') as outfile:
        csv_writer = csv.writer(outfile)
        for _r in rows:
            csv_writer.writerow(_r)

def resolve_args(selector, *dicts, default=None, comp=lambda _x, _y: _x is _y, 
    unsetter=None, omit=True):
    '''
    convenient function to resolve parameters by merging multiple 
    dictionaries and select in only relevant items

    basic idea: the value from the first dictionary that is not missing
    and does not equal to the default value is placed in the output
    
    arguments:
     - selector: an iterable used to select keys in the dictionary
     - *dicts: each remaining positional argument should be a dictionary
    
    keyword-only arguments:
     - default: the default value for which signal that the resolution
            of value is to be passed downstream
     - comp: the comparison used to check if a given value is equivalent
            to the default
     - unsetter: a value used to _affirmatively_ assign the default value
            to a key, overriding any down-stream non-default values. Must
            compare to false with `default` to be in effect
     - omit: if True, keys that resolve to the default value is omitted
            in the returned dict

    returns: a python dict
    '''

    outdict = { _k: default for _k in selector }

    for _d in dicts:
        for _k in selector:
            val = outdict[_k]
            outdict[_k] = _d.get(_k, default) if comp(val, default) else val

    if not comp(unsetter, default):
        for _k in selector:
            val = outdict[_k]
            outdict[_k] = default if (val==unsetter) else val

    if omit:
        outdict = {
            _k: _v for _k, _v in outdict.items() if not comp(_v, default)
        }
    
    return outdict

def resolve_arg(name, *dicts, default=None, comp=lambda _x, _y: _x is _y, 
    unsetter=None):
    '''
    convenient function to resolve a parameter by merging values from 
    multiple dictionaries 

    basic idea: the value from the first dictionary that is not missing
    and does not equal to the default value is placed in the output
    
    arguments:
     - name: the name of the parameter to be extracted
     - *dicts: each remaining positional argument should be a dictionary
    
    keyword-only arguments:
     - default: the default value for which signal that the resolution
            of value is to be passed downstream
     - comp: the comparison used to check if a given value is equivalent
            to the default
     - unsetter: a value used to _affirmatively_ assign the default value
            to a key, overriding any down-stream non-default values. Must
            compare to false with `default` to be in effect

    returns: a python dict
    '''

    val = default

    for _d in dicts:
        val = _d.get(name, default) if comp(val, default) else val

    if not comp(unsetter, default):
        val = default if (val==unsetter) else val
    
    return val

__all__ = [
    "QuestionInfo", "QuestionSet", "PrePostPair", "PairOptimizer", 
    "read_question_info", "write_csv"
]

if __name__=="__main__":

    import argparse, json, datetime

    defaults = {
        "qNum": 15,
        "typeMin": 5,
        "diffMax": 2,
        "TMin": 0.4,
        "TMax": 0.6,
        "stepMax": 100,
        "badMax": 50,
        "badSeqMax": 10,
        "effort": 0.5,
        "verbosity": 1,
        "n_runs": 3,
        "header_row": 2,
        "label_idx": 0,
        "count_idx": 1,
        "TF_idx": 2,
        "type_idx": 4,
        "num_type": 5,
        "subtype_idx": 9,
        "num_subtype": 15,
        "write_mode": "w"
    }

    opt_params = [
        "qNum", "typeMin", "diffMax", "TMin", "TMax", "stepMax", 
        "badMax", "badSeqMax", "effort"
    ]

    read_params = [
        "header_row", "label_idx", "count_idx", "TF_idx", 
        "type_idx", "num_type", "subtype_idx", "num_subtype", 
    ]

    # special case for option -j
    class write_json(argparse.Action):
    
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, parser, namespace, values, option_string):
            if values is not None:
                with open(values, "w") as outfile:
                    json.dump(defaults, outfile, indent = 4)
                print("defaults written to {}".format(values))
                parser.exit()

    # special case for option -H
    class print_docstring(argparse.Action):
    
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, parser, namespace, values, option_string):
            print(__doc__)
            parser.exit()

    # process command-line options
    parser = argparse.ArgumentParser(description='genbiomaps auto assign')
    parser.add_argument('input', help='name/path of the question database .csv file')
    parser.add_argument('output', help="name/path of the output .csv file")
    parser.add_argument('-c', '--config', type=str, default=None,
        help='use json file CONFIG to specify options')
    parser.add_argument('-v', '--verbosity', type=int, default=None,
        help='verbosity of log (integer between 0 and 3 inclusive)'
    )
    parser.add_argument(
        '-j', '--json', type=str, default=None, action=write_json,
        help='write the default options to file JSON, then exit'
    )
    parser.add_argument('-H', '--fullhelp', nargs=0, action=print_docstring,
        help='print the full docstring of this python script and exit'
    )

    args = parser.parse_args()
    args_dict = vars(args)

    # read config from json
    if args.config is not None:
        with open(args.config) as infile:
            jargs = json.load(infile)
    else:
        jargs = dict()

    # resolve verbosity
    verbosity = resolve_arg("verbosity", args_dict, jargs, defaults)

    # read question database
    kwargs = resolve_args(read_params, jargs, defaults)
    question_db = read_question_info(args.input, **kwargs)
    if verbosity:
        print("\nQuestion database read from {}\n".format(args.input))

    # extract number of runs:
    n_runs = resolve_arg("n_runs", jargs, defaults)

    # initialize optimizer
    kwargs = resolve_args(opt_params, jargs, defaults)
    optimizer = PairOptimizer(question_db, **kwargs, verbosity=verbosity)

    # run optimizer
    if n_runs < 2:
        optimizer.run()
    else:
        optimizer.run_multiple(n_runs)

    best = optimizer.best
    pre, post = optimizer.export_best(label=False)

    # print and export result
    if verbosity:
        print("")
        if verbosity==2:
            print("pre: {}".format(pre))
            print("post: {}".format(post))
        
        print("Number of sub-parts: {}".format(best.tally_subparts()))
        print("Proportion of T's: ({:.2f}, {:.2f})".format(
            *best.calc_T_ratio()
        ))

        if verbosity > 1:
            print("Category counts:")
            tmp = best.tally_types()
            print("pre: {}".format(dict(tmp[0])))
            print("post: {}".format(dict(tmp[1])))
            
            print("Sub-category counts:")
            tmp = best.tally_subtypes()
            print("pre: {}".format(dict(tmp[0])))
            print("post: {}".format(dict(tmp[1])))
    
    write_mode = resolve_arg("write_mode", args_dict, jargs, defaults)
    write_word = "appended" if (write_mode=="a") else "written"
    now = datetime.datetime.today().strftime(r"%m/%d/%Y %H:%M:%S")
    header = ["Timestamp", now]
    write_csv(args.output, header, ["pre"] + pre, ["post"] + post, 
        write_mode=write_mode)

    if verbosity:
        print("\nAssigned pair {} to {}\n".format(write_word, args.output))