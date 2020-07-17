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
eq_names = {
    "wf": ["f0", "f1", "f2"],
    "pot": ["vi", "vij"],
    "op": ["op0", "op1", "op2"],
    "supp": ["ri", "rij", "rcm"]
}

def brackets(expr):
    return r"\{" + expr + r"\}"



for name in ["wf", "pot"]:
    @app.callback(
        Output(f"{name}-config-vars", "expression"),
        [Input(f"vars.{name}.{eq_name}", "data") eq_name in eq_names(name)]
    )
    def config_vars(*list_data_vars):
        if any(data_vars["state"] == "failed" for data_vars in list_data_vars):
            raise PreventUpdate
        
        # returns something like "{{a, b, c}, {d, e, f}}"
        return brackets(
            ",".join(
                brackets(
                    ",".join(
                        latex(var) 
                        for var in data_vars["value"])
                    ) 
                for data_vars in list_data_vars
            )
        )
               




other_equations = set(eq_names.keys()) - set(["pot","wf"])

@app.callback(
    Output("other-config-vars", "expression"),
    [Input(f"vars.{name}.{eq_name}", "data") for name in other_equations for eq_name in eq_names(name)]
)
def other_config_vars(*list_data_vars):

    return brackets(
            ",".join(
                brackets(
                    ",".join(
                        latex(var) 
                        for var in data_vars["value"])
                    ) 
                for data_vars in list_data_vars
            )
        )


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

@app.callback(
    [Output("interval-sim","disabled"),Output("compiled", "data"),Output("compiled", "data")]
    [Input("run-sim", "n_clicks")],
    [State("compiled", "data"), State("config","data")]

)
def tdvm_run(n, data_compiled, data_config):
    return True, 
    var_exprs_wf, var_names_wf  = get_vars(mem_vars, ["wf"])
    var_exprs_pot, var_names_pot  = get_vars(mem_vars, ["pot"])

    vars_wf = mem_vars["wf"]
    vars_pot = mem_vars["pot"]
    try:
        wf_config = {{var_name: var for var_name, var in zip(var_names, vars)} 
                     for var_names, vars in  zip(var_names_wf, vars_wf)}
        pot_config = {{var_name: var for var_name, var in zip(var_names, vars)} 
                      for var_names, vars in  zip(var_names_pot, vars_pot)}
    except:
        return data
        raise PreventUpdate("Missmatched configuration values for wave-function or potential variables")

    if not is_running:
        mem_tdvm["is_running"] = True
        mem_tdvm["curr_time"] = 0.0
        data["El"], data["El_Op_dc_iter"], data["Ev_Matesq_Vdret"] = compute_funcs(config)
    return data



@app.callback(
    [Output("mem-tdvm", "data"), Output("simulation-progres", "value"), Output("simulation-progres", "color")],
    [Input("10-interval", "n_times")],
    [State("mem-tdvm", "data")]
)
def tdvm_iteration(mem_tdvm):
    if not mem_tdvm.get("is_running", False): raise PreventUpdate

    wf_config = mem_tdvm["config wf"]
    pot_config = mem_tdvm["config pot"]
    mem_tdvm["t"] = mem_tdvm.get("t", [])
    mem_tdvm["c"] =  mem_tdvm.get("c", [])
    curr_time = mem_tdvm["curr_time"]
    delta = mem_tdvm["Dt"]
    config = {"Dh": data_vars["Dh"], 
              "Dt": data_vars["Dt"],
              "n_it": data_vars["n_it"],
              "p": tuple(tuple(value for value in eq_config.values()) 
                         for eq_config in wf_config.values()),
              "c": tuple(tuple(value for value in eq_config.values()) 
                         for eq_config in pot_config.values())}
    mem_tdvm["t"].append(curr_time)


    ev,Matesq, Vdret= Ev_Matesq_Vdret(**config)
    cdot = numpy.linalg.pinv(Matesq) @ Vdret
    c0s, cis, cijs = c
    c_arr = np.array([*c0s, *cis, *cijs])
    c_arr -= cdot * delta
    c_arr = np.clip(c_arr, a_min = [0.0]*len(c_arr), a_max=[10e+10]*len(c_arr))
    splits = np.cumsum(list(map(len,c[:-1])))
    c = tuple(list(np.split(c_arr, splits)))
    curr_time += delta
    mem_tdvm["curr_time"] = curr_time
    mem_tdvm["curr_time"].append(c)
    if curr_time == mem_tdvm["sim_sec"]: return  mem_tdvm, 100, "success"
    else: return mem_tdvm, int(curr_time / data["sim_sec"] * 100), "primary"

from functools import lru_cache

@lru_cache
def parser_nested_expr():
    from pyparsing import Forward, Word, alphas, nestedExpr, pyparsing_common, Suppress
    enclosed = Forward()
    nested_brackets = nestedExpr("{", "}", content=enclosed)
    enclosed << (pyparsing_common.real | pyparsing_common.integer | Suppress(",") | nested_brackets)
    return lambda expr: enclosed.parseString(expr).asList()[0]

def parse_nested_expr(expr):
    parser = parser_nested_expr()
    return parser(expr)
@app.callback(Output("value.Dt","data"), 
              [Input("Dh-value","value")],
              [State("value.Dtm", "data")])
def assign_Dh(Dh, data_dh):
    data_dh["value"] = Dh
    data_dh["state"] = "success"
    return data_dh

@app.callback(Output("value.Dt", "data"), 
              [Input("Dt-value","value")],
              [State("value.Dt", "data")])
def assign_Dt(Dt, data_dt):
    data_dt["value"] = Dt
    data_dh["state"] = "success"
    return data_dt

@app.callback(Output("value.n_it","data"), 
              [Input("it-value","value")],
              [State("value.n_it", "data")])
def assign_n_it(n_it, mem_tdvm):
    data_n_it["value"] = n_it
    data_n_it["state"] = "success"
    return data

@app.callback(Output("value.sim_sec","data"),
             [Input("sec-value","value")],
             [State("value.sim_sec", "data")])
def assign_sim_sec(sim_sec, data_sim_sec):
    data_sim_sec["value"] = sim_sec
    data_sim_sec["state"] = "success"
    return data



from itertools import zip_longest

for name in ["wf", "pot"]:
    @app.callback(
        [Output("config", "data")],
        [Input(f"{name}-config-vars","value")],
        [State("config", "data"), State(f"vars.{name}.{name_eq}") for name_eq in eq_names[name]]
    )
    def assign_eq_val(expr, data_config, *datas):
        value_vars = parse_nested_expr(expr)
        name_vars = [[var.name for var in data["value"]] for data in datas]
        var_names, _ = get_vars(mem_vars, eq_name)
        sentinel = object()
        for [values, names] in zip_longest([value_vars, name_vars], fillvalue=sentinel):
            if sentinel not in [values, names]:
                for pair in zip_longest([values, names], fillvalue=sentinel):
                    if sentinel in pair:
                        data_config["value"] = None
                        data_config["state"] = "failed"
                        return data_config
            else:
                data_config["value"] = None
                data_config["state"] = "failed"
                return data_config

        data_config["value"] = value_vars
        data_config["state"] = "success"

        return data_config

@app.callback([Output("start-sim", "disabled"),Output("spinner-sim","children")]
              [Input("value.Dt", "data"), Input("value.Dh", "data"), Input("value.n_it", "data"), Input("value.sim_sec", "data"),
               Input("run-sim", "n_clicks")])
def disable_button_start_sim(*data_values):
    all_defined = all(data["state"] == "success" for data in data_values)
    ctx = dash.callback_context
    is_clicked = any(comp["prop_id"] == "run-sim.n_clicks" for comp in ctx.triggered)
    return not all_defined or is_clicked, not is_clicked 








controls_tdvm = dbc.Card(
    [
        dcc.Store(id="data.Dh", data={"value":None,"status":"failed"}),
        dcc.Store(id="data.Dt", data={"value":None,"status":"failed"}),
        dcc.Store(id="data.n_it", data={"value":None,"status":"failed"}),
        dcc.Store(id="data.sim_sec", data={"value":None,"status":"failed"}),
        dbc.FormGroup(
            [
                render_latex(expression=r"Dh"),
                dbc.Input(id="Dh-value", type="number", value=1/2)
            ]
        ),
        dbc.FormGroup(
            [
                render_latex(expression=r"\Delta t"),
                dbc.Input(id="Dt-value", type="number", value=0.01),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Wave-Function Configuration"),
                dash_katex.DashKatex(expression="",
                    displayMode=True,
                    throwOnError=False,
                    id="wf-config-vars"
                ),
                dbc.Input(id="wf-config-vars-value", type="text"),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Potential Configuration"),
                dash_katex.DashKatex(expression="",
                    displayMode=True,
                    throwOnError=False,
                    id="pot-config-vars"
                ),
                dbc.Input(id="pot-config-vars-value", type="text"),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Other Vars Configuration"),
                dash_katex.DashKatex(expression="",
                    displayMode=True,
                    throwOnError=False,
                    id="other-config-vars"
                ),
                dbc.Input(id="other-config-vars-value", type="text"),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Number of iterations for Metropolis algorithm (higher leads to more accurate results)"),
                dbc.Input(id="it-value", type="number", value=1000),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Seconds of simulation"),
                html.Div(id="other-config"),
                dbc.Input(id="sec-value", type="number", value=1.0),
            ]
        ),
        dbc.Button("Run simulation", color="primary", id="run-sim",block=True),

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
            controls_tdvm,

        ]
    ),
    className="p-5",
)


if __name__ == "__main__":
    app.run_server(debug=True)
