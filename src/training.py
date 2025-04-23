

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    hello = "hello, world"
    print(hello)
    return


if __name__ == "__main__":
    app.run()
