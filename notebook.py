# /// script
# dependencies = [
#     "anthropic==0.78.0",
#     "graphviz==0.21",
#     "marimo",
#     "pydantic-ai-slim==1.56.0",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import q1.week4 as w
    import random

    return mo, random, w


@app.cell
def _(random, w):
    random.seed(142)
    TRAIN_DATA = [
        ([-2.0, -1.0], 0),
        ([-1.5, -1.2], 0),
        ([-2.2, -0.8], 0),
        ([0.0, 2.0], 1),
        ([0.5, 1.8], 1),
        ([-0.5, 2.2], 1),
        ([2.0, -1.0], 2),
        ([1.5, -1.3], 2),
        ([2.2, -0.7], 2),
    ]

    xs = [x for x, _ in TRAIN_DATA]
    ys = [y for _, y in TRAIN_DATA]

    layers = [w.create_layer(2, 4), w.create_layer(4, 4, non_lin="None")]
    for layer in layers:
        print(layer)
        print(layer.neurons)
        print()
    mlp = w.MLP(layers)
    return mlp, xs, ys


@app.cell
def _():
    import graphviz

    return (graphviz,)


@app.cell
def _(graphviz, w):
    def trace(root: w.Value) -> tuple[set[w.Value], set[tuple[w.Value, w.Value]]]:
        """Build a set of all nodes and edges in the computation graph."""
        nodes: set[w.Value] = set()
        edges: set[tuple[w.Value, w.Value]] = set()

        def build(v: w.Value) -> None:
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges


    def draw_graph(root: w.Value, format: str = "svg", rankdir: str = "LR") -> graphviz.Digraph:
        """
        Render the full computation graph of a Value node using Graphviz.
        Shows data and grad for every node, and the operation that produced it.
        """
        nodes, edges = trace(root)
        dot = graphviz.Digraph(format=format, graph_attr={"rankdir": rankdir, "bgcolor": "white"})

        for n in nodes:
            label_str = n.label if hasattr(n, "label") and n.label else ""
            grad_val = f"{n.grad:.4f}" if hasattr(n, "grad") else "N/A"
            node_label = f"{{ {label_str} | data: {n.data:.4f} | grad: {grad_val} }}"
            dot.node(
                name=str(id(n)),
                label=node_label,
                shape="record",
                style="filled",
                fillcolor="#e8f4fd",
            )
            # If the node has an op, create an intermediate op node
            if hasattr(n, "op") and n.op:
                op_id = str(id(n)) + n.op
                dot.node(op_id, label=n.op, shape="circle", style="filled", fillcolor="#ffeeba", width="0.3")
                dot.edge(op_id, str(id(n)))
                for child in n._prev:
                    dot.edge(str(id(child)), op_id)
            # fallback: just connect prev -> n
            elif n._prev:
                for child in n._prev:
                    dot.edge(str(id(child)), str(id(n)))

        return dot

    return (draw_graph,)


@app.cell
def _(graphviz, w):
    def draw_network_architecture(mlp: w.MLP) -> graphviz.Digraph:
        """
        Draw a high-level architecture diagram of the MLP showing
        each neuron, its weights, and bias with data + grad values.
        """
        dot = graphviz.Digraph(format="svg", graph_attr={
            "rankdir": "LR",
            "bgcolor": "white",
            "nodesep": "0.5",
            "ranksep": "1.5",
        })

        # --- Input layer ---
        n_inputs: int = len(mlp.layers[0].neurons[0].weights) if mlp.layers else 0
        with dot.subgraph(name="cluster_input") as c:
            c.attr(label="Input", style="dashed", color="grey")
            for i in range(n_inputs):
                c.node(f"input_{i}", label=f"x[{i}]", shape="circle",
                       style="filled", fillcolor="#d4edda", width="0.6")

        # --- Hidden / output layers ---
        for li, layer in enumerate(mlp.layers):
            is_last = li == len(mlp.layers) - 1
            layer_label = "Output" if is_last else f"Layer {li}"
            with dot.subgraph(name=f"cluster_layer_{li}") as c:
                c.attr(label=layer_label, style="dashed", color="grey")
                for ni, neuron in enumerate(layer.neurons):
                    bias_grad = f"{neuron.bias.grad:.4f}" if hasattr(neuron.bias, "grad") else "N/A"
                    neuron_label = (
                        f"n[{li},{ni}]\\n"
                        f"bias: {neuron.bias.data:.4f}\\n"
                        f"b.grad: {bias_grad}"
                    )
                    fill = "#fff3cd" if is_last else "#e8f4fd"
                    c.node(f"neuron_{li}_{ni}", label=neuron_label,
                           shape="box", style="filled,rounded", fillcolor=fill)

            # Edges from previous layer (or inputs) to this layer
            for ni, neuron in enumerate(layer.neurons):
                for wi, weight in enumerate(neuron.weights):
                    w_grad = f"{weight.grad:.4f}" if hasattr(weight, "grad") else "N/A"
                    edge_label = f"w={weight.data:.3f}\ng={w_grad}"
                    if li == 0:
                        src = f"input_{wi}"
                    else:
                        src = f"neuron_{li-1}_{wi}"
                    dot.edge(src, f"neuron_{li}_{ni}", label=edge_label, fontsize="9")

        return dot

    return (draw_network_architecture,)


@app.cell
def _(mo):
    mo.md(f"""
    ## Neural Network Visualisation

    Use the tabs below to explore:

    - **Architecture** – high-level view of every layer, neuron, weight and gradient.
    - **Computation Graph** – the full autograd graph for a single forward pass.

    > **Tip:** Run a forward pass (and optionally a backward pass) before rendering
    > so that `data` and `grad` values are populated.
    """)
    return


@app.cell
def _(draw_graph, mlp, w, xs, ys):
    # Run a forward pass so the graph is populated
    _sample_x: list[w.Value] = [w.Value(data=v, label=f"x{i}", prev=[]) for i, v in enumerate(xs[0])]
    outputs: list[w.Value] = mlp(_sample_x)

    # Attempt a backward pass to populate gradients
    # Build a simple loss for demonstration (target class 0)
    _target_class: int = ys[0]
    _loss = w.Value(data=0.0, label="loss", prev=[])

    # softmax-style: we want to maximise the score of the correct class
    # simple svm-like hinge loss per pair
    _losses: list[w.Value] = []
    for _i, _out in enumerate(outputs):
        if _i != _target_class:
            _margin = _out - outputs[_target_class] + w.Value(data=1.0, label="margin", prev=[])
            # relu of margin
            _relu_margin = _margin.relu() if hasattr(_margin, "relu") else _margin
            _losses.append(_relu_margin)

    if _losses:
        _total_loss = _losses[0]
        for _l in _losses[1:]:
            _total_loss = _total_loss + _l
        _total_loss.label = "total_loss"
        # backward
        if hasattr(_total_loss, "backward"):
            _total_loss.backward()
        loss_graph = draw_graph(_total_loss)
    else:
        loss_graph = draw_graph(_outputs[0])
    return loss_graph, outputs


@app.cell
def _(
    draw_graph,
    draw_network_architecture,
    graphviz,
    loss_graph,
    mlp,
    mo,
    outputs: "list[w.Value]",
):
    _arch_diagram: graphviz.Digraph = draw_network_architecture(mlp)
    _comp_graph: graphviz.Digraph = draw_graph(outputs[0])

    mo.ui.tabs({
        "🏗️ Architecture": mo.Html(_arch_diagram.pipe(format="svg").decode("utf-8")),
        "🔀 Computation Graph (single output)": mo.Html(_comp_graph.pipe(format="svg").decode("utf-8")),
        "📉 Loss Computation Graph": mo.Html(loss_graph.pipe(format="svg").decode("utf-8")),
    })
    return


if __name__ == "__main__":
    app.run()
