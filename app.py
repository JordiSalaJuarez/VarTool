import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import plotly.graph_objs as go
import dash_katex
from functools import partial
from collections import defaultdict
from sympy import latex
import tdvmc.mathematica_to_sympy as m2s
from dash.exceptions import PreventUpdate

# app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

def tree(): return defaultdict(tree)

config = tree()

def generate_input_equation(app,
                            eq_expr, 
                            id_eq,
                            id_var,
                            addr_mem_vars,
                            placeholder_eq="write your equation", 
                            placeholder_var="put variables comma separated"):
    id_store_vars =  ".".join(["vars", *addr_mem_vars])
    id_store_eq = ".".join(["eq", *addr_mem_vars])
    render_latex = partial(dash_katex.DashKatex, displayMode=False, throwOnError=False)

    @app.callback(
        [Output(id_store_eq, 'data')],
        [Input(id_eq, "value")],
        [State(id_store_vars, "data")]
    )
    def store_eq(expr, data_eq ,data_vars):
        try:
            data_eq["value"] = m2s.parse(text, vars={var.name: var for var in data_vars["value"]})
            data_eq["state"] = "success"
            return data_eq
        except Exception as e:
            data_eq["value"] = None
            data_eq["state"] = "failed"
            data_eq["msg"] = str(e) 
            return data_eq

    @app.callback(
        [Output(id_store_vars,"data")],
        [Input(id_vars, "value")],
        [State(id_store_vars,"data")]
    )
    def store_vars(expr, data_vars):
        try:
            variables = parse_nested_expr(expr)
            name, subname, *_ = mem_var
            data_vars["value"] = [m2s.parse(var) for var in variables]
            data_vars["state"] = "success"
            return data_vars
        except Exception as e:
            data_vars["value"] = None
            data_vars["state"] = "failed"
            return data_vars

    @app.callback(
        [Output(id_var, "valid"), Output(id_var, "invalid")],
        [Input(id_store_vars,"data")],
    )
    def check_valid_vars(data_vars):
        is_valid = data_vars["state"] == "success"
        return is_valid, not is_valid

    @app.callback(
        [Output(f"{id_var}-list", "expression")],
        [Input(id_store_vars,"data")],
    )
    def show_vars(data_vars):
        if data_vars["state"] == "failed": raise PreventUpdate
        else: 
            variables = data_vars["value"]
            return  r"\{"+ ", ".join(latex(var) for var in variables) + r"\}"

    @app.callback(
        [Output(id_eq, "valid"), Output(id_eq, "invalid")],
        [Input(id_store_eq,"data")],
    )
    def check_valid_vars(data_eq):
        is_valid = data_eq["state"] == "success"
        return is_valid, not is_valid

    @app.callback(
        [Output(f"{id_eq}-latex", "expression")],
        [Input(id_store_eq,"data")],
    )
    def show_eq(data_eq):
        if data_eq["state"] == "failed": raise PreventUpdate
        else: return data_eq["value"]
            return latex(data_vars["value"])

    @app.callback(
        [Output(f"{id_eq}-popover", "is_open"), Output(f"{id_eq}-popover-body", "children")],
        [Input(id_store_eq,"data")]
    )
    def show_error_eq(data_eq):
        msg = data_eq.get("msg", "")
        has_failed = data_eq["state"] == "failed"
        return has_failed, msg
    
    return dbc.Card(
        [
            dcc.Store(id=id_store_eq, data={"value":None,"status":"failed"}),
            dcc.Store(id=id_store_vars, data={"value":[],"status":"success"}),
            dbc.CardBody(
                [
                    dbc.FormGroup(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Container(render_latex(expression=eq_expr)), 
                                        width=3
                                    ),
                                    dbc.Col(
                                        dbc.Input(
                                            type="text", 
                                            id=id_eq, 
                                            placeholder=placeholder_eq
                            ))], align="center")
                            
                            ,
                            dbc.Popover(
                                    [
                                        dbc.PopoverHeader("Error"),
                                        dbc.PopoverBody("And here's some amazing content. Cool!", id=f"{id_eq}-popover-body"),
                                    ],
                                    id=f"{id_eq}-popover",
                                    is_open=False,
                                    target=id_eq,
                            ),
                        ]
                    ),
                    dbc.Container(render_latex(expression="",
                        id=f"{id_eq}-latex"
                    ))
                    ,
                ]
            ), 
            dbc.CardFooter(
                [
                    dbc.FormGroup(
                        [
                            html.H5("Variables", className=f"{id_vars}-test"),
                            dbc.Input(
                                type="text", id=id_vars, placeholder=placeholder_var
                            ),
                            html.Br()
                            ,
                            dbc.Container(render_latex(expression="",
                                id=f"{id_vars}-list",
                            ))
                        ],
                        style={'width': "auto"}
                    )
                ]
            )
        ],
        style={"width": "30rem"}
    )

render_latex = partial(dash_katex.DashKatex, displayMode=False, throwOnError=False)


var_ids = {
    "utils": {"r_i": "r_i-var", "r_ij": "r_ij-var", "r_cm": "r_cm-var"} ,
    "wf": {"f0": "f0-var", "f1": "f1-var", "f2": "f2-var"}, 
    "pot": {"vi": "vi-var", "vij": "vij-var"},
    "op": {"op0": "op0-var", "op1": "op1-var" , "op2": "op2-var"}
}
}



card_utils = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H3("Helpers", className="op"),
                html.Div(html.P("Define helper functions"),style={'display': 'inline-block'}),
                dbc.CardDeck(
                    [                       
                        generate_input_equation(app=app, eq_expr=r"r_i", id_eq="r_i", id_var=var_ids["utils"]["r_i"], addr_mem_vars=("utils", "r_i") ),
                        generate_input_equation(app=app, eq_expr=r"r_{ij}", id_eq="r_ij", id_var=var_ids["utils"]["r_ij"], addr_mem_vars=("utils", "r_ij")),
                        generate_input_equation(app=app, eq_expr=r"r_\text{cm}", id_eq="r_cm", id_var=var_ids["utils"]["r_cm"], addr_mem_vars=("utils", "r_cm")),
                    ]
                ),
            ]
        ),
        dbc.CardFooter( html.H5("Variables Potential Energy", className="vars"))
    ]
)

card_wave_function = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H3("Wave Function", className="wave-function"),
                html.P("Define the wave funtion Jastrow terms"),
                dbc.CardDeck(
                    [                       
                        generate_input_equation(app=app, eq_expr=r"f_0", id_eq="f0", id_var=var_ids["wf"]["f0"], addr_mem_vars=("wf", "f0")),
                        generate_input_equation(app=app, eq_expr=r"f_1(i)", id_eq="f1", id_var=var_ids["wf"]["f1"], addr_mem_vars=("wf", "f1")),
                        generate_input_equation(app=app, eq_expr=r"f_2(i,j)", id_eq="f2", id_var=var_ids["wf"]["f2"], addr_mem_vars=("wf", "f2"))
                    ]
                ),
            ]
        ),
        dbc.CardFooter( html.H5("Variables Wave Function", className="vars"))
    ]  
)



card_potential = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H3("Potential Energy", className="pot-energy"),
                html.Div(html.P("Define potential by"),style={'display': 'inline-block'}),
                html.Div(render_latex(expression=r"\; v_i,\;"),style={'display': 'inline-block'}),
                html.Div(render_latex(expression=r"v_{ij}"),style={'display': 'inline-block'}),
                dbc.CardDeck(
                    [                       
                        generate_input_equation(app=app, eq_expr=r"v_i", id_eq="vi", id_var=var_ids["pot"]["vi"], addr_mem_vars=("pot", "vi")),
                        generate_input_equation(app=app, eq_expr=r"v_{ij}", id_eq="vij", id_var=var_ids["pot"]["vij"], addr_mem_vars=("pot", "vij")),
                    ]
                ),
            ]
        ),
        dbc.CardFooter( html.H5("Variables Potential Energy", className="vars"))
    ]
)

card_operators = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H3("Operators", className="op"),
                html.Div(html.P("Define operators"),style={'display': 'inline-block'}),
                html.Div(render_latex(expression=r"\; \mathcal{O}_0,\;"),style={'display': 'inline-block'}),
                html.Div(render_latex(expression=r"\; \mathcal{O}_1,\;"),style={'display': 'inline-block'}),
                html.Div(render_latex(expression=r"\; \mathcal{O}_2,\;"),style={'display': 'inline-block'}),
                dbc.CardDeck(
                    [                       
                        generate_input_equation(app=app, eq_expr=r"\mathcal{O}_0", id_eq="op0", id_var=var_ids["op"]["op0"], addr_mem_vars=("op", "op0")),
                        generate_input_equation(app=app, eq_expr=r"\mathcal{O}_1(i)", id_eq="op1", id_var=var_ids["op"]["op1"], addr_mem_vars=("op", "op1")),
                        generate_input_equation(app=app, eq_expr=r"\mathcal{O}_2(i,j)", id_eq="op2", id_var=var_ids["op"]["op2"], addr_mem_vars=("op", "op2")),
                    ]
                ),
            ]
        ),
        dbc.CardFooter( html.H5("Variables Potential Energy", className="vars"))
    ]
)

controls_tdmb = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Dh"),
                dcc.Input(id="Dh-value", type="number", value=1/2)
            ]
        ),
        dbc.FormGroup(
            [
                render_latex(expression=r"\Delta t"),
                dcc.Input(id="Dt-value", type="number", value=0.01),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Potential Configuration"),
                html.Div(id="pot-config"),
                dash_katex.DashKatex(expression="",
                    displayMode=True,
                    throwOnError=False,
                    id="pot-config-vars"
                )
            ]
        ),

    ],
    body=True,
)



config_tdvm = {"fails": False}


# @app.callback(
#     Output(id_var, "valid"), Output(id_var, "invalid")
#     [Input(id_var, "value") for id_var var_ids["potential"].values()]
# )
# def update_input_potential(*args):
#     [config[id_var] for id_var var_ids["potential"].values()]






app.layout =  dbc.Container(
    html.Div(
        [
            dcc.Store(id="mem-eqs"),
            dcc.Store(id="mem-vars"),
            html.H2("Variational Montecarlo Equations"),
            dbc.Card(
                [
                    dbc.CardHeader(
                        dcc.Upload(
                            [
                                'Drag and Drop or ',
                                html.A('Select a File', href="#")
                            ], 
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '0.5px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center'
                            }
                        )
                    ),
                    dbc.CardBody(
                        [
                            card_utils,
                            html.Br(),
                            card_wave_function,
                            html.Br(),
                            card_potential,
                            html.Br(),
                            card_operators
                            
                        ]            
                    )
                ]
            ),
            html.Hr(),
            html.H2("Time Dependend Variational Montecarlo"),

        ]
    ),
    className="p-5",
)


if __name__ == "__main__":
    app.run_server(debug=True)
