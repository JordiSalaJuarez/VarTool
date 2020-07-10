#
# This example shows how to write a basic calculator with variables.
#

from lark import Lark, Transformer, v_args
from functools import reduce
# from sympy import IndexedBase, Mul, Pow, Add, Symbol, Function, Product, Sum
from sympy import *
import sympy

try:
    input = raw_input   # For Python2 compatibility
except NameError:
    pass


calc_grammar = """
    ?start:  expr
        | NAME "=" expr -> assig

    ?expr: plus_substract

    ?plus_substract: times_divide
        | plus_substract "+" times_divide -> plus
        | plus_substract "-" times_divide -> substract

    ?times_divide: power
        | times_divide "*" power -> times
        | times_divide "/" power  -> divide
    
    ?power: atom
        | power "^" atom -> power

    ?atom: NUMBER -> number
         | call
         | list
         | indexed_symbol  
         | symbol
         | "+" atom
         | "-" atom -> neg
         | atom "//" NAME -> simplification
         | "(" expr ")"
         

    ?call: "Sqrt" "[" expr "]" -> sqrt
         | "Log" "[" expr "]" -> ln
         | "Log" "[" expr "," expr "]" -> log
         | "Sum" "[" expr ("," expr )*  "]" -> sum
         | "Product" "[" expr ("," expr )*  "]" -> product
         | "Power" "[" expr "," expr "]" -> power
         | "Times" "[" expr ("," expr )*  "]" -> times 
         | "Plus" "[" expr ("," expr )* "]" -> plus
         | "Substract" "[" expr "," expr "]" -> substract
         | "Divide" "[" expr "," expr "]" -> divide
         | "Function" "[" expr "," expr "]" -> function


    ?list: "{" ((expr ",")* expr)? "}" -> tuple

    ?symbol: "E" -> e
           | "\\[" NAME "]" -> symbol
           | NAME       -> symbol

    ?indexed_symbol: NAME "[" (expr ",")* expr "]" -> indexed_symbol


    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
""" 
simp_types = {"Simplify": simplify, "Factor": factor, "Cancel": cancel, "Expand": expand}

@v_args(inline=True)    # Affects the signatures of the methods
class CalculateTree(Transformer):
    from operator import neg
    from sympy.abc import alpha as _alpha , beta as _beta , gamma as _gamma
    from sympy import Symbol, Number as number, Tuple as tuple, Mul as times, Add as plus, Pow as power, E, Lambda as function
    e, alpha , beta, gamma = (lambda self,var=var: var for var in [E, _alpha, _beta, _gamma])
    def assig(self, var, expr): self.vars[var] = expr
    def indexed_symbol(self, var, *args):
        if var in self.vars:
            if callable(self.vars[var]): return self.vars[var](*args,)
            else: return self.vars[var][(*args,)]
        else: return IndexedBase(var)[(*args,)]
    sum = lambda self, f, *args : Sum(f, *args)
    product = lambda self, f, *args : Product(f, *args)
    divide = lambda self, x, y: Mul(x, Pow(y, -1))
    substract = lambda self, x, y: Add(x, Mul(-1, y))
    symbol = lambda self, var: self.vars.get(var, Symbol(var, real=True))
    
    def sqrt(self, expr): return sympy.sqrt(expr)
    def ln(self, expr): return sympy.ln(expr)
    def log(self, expr, base): return sympy.log(expr, base)
    
    def simplification(self, expr, name):
        return simp_types.get(name, simplify)(expr)

    

    def call(self, fun, *args):
        return vars.get(fun, Function(fun))(*args)
    def __init__(self):
        self.vars = {}


parser = Lark(calc_grammar, parser='lalr', transformer=CalculateTree())

def parse(text, vars=None, start=None):
    parser.options.transformer.vars = {} if vars is None else vars
    return parser.parse(text,start=None).doit()



if __name__ == '__main__':
    print(parser.parse(input()))