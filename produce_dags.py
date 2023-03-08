import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import graphviz 
from sklearn.decomposition import PCA


# Import Data
data = pd.read_csv("a1_data.csv")


# One-tailed F-test
def f_test(X1: pd.Series, X2: pd.Series):
    f = np.var(X1, ddof=1)/np.var(X2, ddof=1)
    nun = X1.size-1
    dun = X2.size-1
    p_value = 1-stats.f.cdf(f, nun, dun)
    return f, p_value


# Method from Lecture 5
class CausalDAG:

    def __init__(self, data, sig:float=0.05):
        self.data = data
        self.variables = list(data.columns)
        self.sig = sig
        a = self.variables
        b = [[] for _ in range(len(a))]
        self.dag = dict(zip(a, b))
        c = self.variables
        d = [[] for _ in range(len(a))]
        self.hierarchical_dag = dict(zip(c, d))

    def test1(self, X1: str, X2: str):
        """
        Check if slope (a) of Linear model X2 = a * X1 + b
        H0 : a == 0 
        H1 : a != 0 
        using Z-statistic

        input  : 
                X1  : independent variable
                X2  : dependent variable
                sig : significance level of statistical test 

        output : 
                True if we reject the null hypothesis
                False if we fail to reject then null hypothesis
        """
        X = sm.add_constant(self.data[X1])
        y = self.data[X2]
        linear_model = sm.OLS(y, X).fit()
        return linear_model.pvalues[1] < self.sig
    
    def test2(self, X1: str, X2: str):
        """
        Check if 
        H0 : var(X2) / var(X1) == 1
        H1 : var(X2) / var(X1) > 1 
        using F-statistic

        input  : 
                X1  : independent variable
                X2  : dependent variable
                sig : significance level of statistical test 

        output : 
                True if we reject the null hypothesis
                False if we fail to reject then null hypothesis        
        """
        f, p_value = f_test(self.data[X2], self.data[X1])
        return p_value / 2 < self.sig
    
        
    def check_causal_connection(self, X1:str, X2:str):
        if self.test1(X1, X2) and self.test2(X1, X2):
            self.dag[X2].append(X1)

    def discover_dag(self):
        for dependent_variable in list(self.dag.keys()):
            for independent_variable in list(self.dag.keys()):
                if dependent_variable != independent_variable:
                    self.check_causal_connection(independent_variable, dependent_variable) 

    def discover_hierarchical_dag(self):
        for parent in list(self.dag.keys()):
            for candidate_child in self.dag[parent]:
                is_direct_child = True
                for causal_connection in self.dag[parent]:
                    if candidate_child != causal_connection:
                        if candidate_child in self.dag[causal_connection]:
                            is_direct_child = False
                if is_direct_child:
                    self.hierarchical_dag[parent].append(candidate_child)
    
    def visualize(self, graph_type: str = "dag"):
        if graph_type == "dag":
            graph = self.dag
        elif graph_type == "hierarchical_dag":
            graph = self.hierarchical_dag
        else:
            return None
        
        dot = graphviz.Digraph(comment='Causal DAG')

        for key in list(graph.keys()):
            dot.node(key, f"Variable {key}")

        for key in list(graph.keys()):
            for value in list(graph[key]):
                dot.edge(f'{key}', f'{value}')

        dot.render(f'graphs/{graph_type}.gv').replace('\\', '/')
        'graphs/doctest-output/round-table.gv.pdf'


# Our method
class CausalDAGalter:

    def __init__(self, data, num_components, sig:float=0.05):

        # Attributes
        self.data = data
        self.variables = list(data.columns)
        self.sig = sig
        self.num_components = num_components

        # Causal DAG Initialization
        a = self.variables
        b = [[] for _ in range(len(a))]
        self.dag = dict(zip(a, b))
        c = self.variables
        d = [[] for _ in range(len(a))]
        self.hierarchical_dag = dict(zip(c, d))

        # Principal Components Loading Matrix
        pca = PCA(n_components=self.num_components)
        pca.fit(data)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        self.loading_matrix = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(num_components)], index=self.variables)
        # Euristic Criterion to handle low factor loadings (won't work for multiple principal components)
        self.loading_matrix[abs(self.loading_matrix) < 0.04 * self.num_components] = 0  

    def test1(self, X1: str, X2: str):
        """
        Check if slope (a) of Linear model X2 = a * X1 + b
        H0 : a == 0 
        H1 : a != 0 
        using Z-statistic

        input  : 
                X1  : independent variable
                X2  : dependent variable
                sig : significance level of statistical test 

        output : 
                True if we reject the null hypothesis
                False if we fail to reject then null hypothesis
        """
        X = sm.add_constant(self.data[X1])
        y = self.data[X2]
        linear_model = sm.OLS(y, X).fit()
        return linear_model.pvalues[1] < self.sig
    
    def test2(self, X1: str, X2: str):
        """
        (1) Representation

            First of all, we represent each vaiable X as a vector
            X_tilda = (corr(X, P1), ... , corr(X, Pn)) , where Pi are 
            the principle components of the given data.

        (2) Order Relation (Causality)

            Based on the above representation we define an order 
            relation which implies causality as described below:

            Xi_tilda < Xj_tilda iff all corresponding coefficients
            of Xi_tilda are smaller than the ones of Xj_tilda

            This will imply Xj ---> Xi 
        """
        return (abs(self.loading_matrix.loc[X1, :]) <= abs(self.loading_matrix.loc[X2, :])).all()
        
    def check_causal_connection(self, X1:str, X2:str):
        if self.test1(X1, X2): 
            if self.test2(X1, X2):
                self.dag[X2].append(X1)

    def discover_dag(self):
        for dependent_variable in list(self.dag.keys()):
            for independent_variable in list(self.dag.keys()):
                if dependent_variable != independent_variable:
                    self.check_causal_connection(independent_variable, dependent_variable) 
    
    def discover_hierarchical_dag(self):
        for parent in list(self.dag.keys()):
            for candidate_child in self.dag[parent]:
                is_direct_child = True
                for causal_connection in self.dag[parent]:
                    if candidate_child != causal_connection:
                        if candidate_child in self.dag[causal_connection]:
                            is_direct_child = False
                if is_direct_child:
                    self.hierarchical_dag[parent].append(candidate_child)

    def visualize(self, graph_type: str = "dag"):
        if graph_type == "dag":
            graph = self.dag
        elif graph_type == "hierarchical_dag":
            graph = self.hierarchical_dag
        else:
            return None
        
        dot = graphviz.Digraph(comment='Causal DAG')
        dot 

        for key in list(graph.keys()):
            dot.node(key, f"Variable {key}")

        for key in list(graph.keys()):
            for value in list(graph[key]):
                dot.edge(f'{key}', f'{value}')

        dot.render(f'graphs/custom_{graph_type}_pc={self.num_components}.gv').replace('\\', '/')
        'graphs/round-table.gv.pdf'


if __name__ == "__main__":

    # Lecture 5 Method
    dag1 = CausalDAG(data)
    dag1.discover_dag()
    dag1.discover_hierarchical_dag()
    dag1.dag
    dag1.visualize(graph_type="dag")
    dag1.hierarchical_dag
    dag1.visualize(graph_type="hierarchical_dag")

    # Custom Method (using different number of PC : theoretically optimal is #PC = 7)
    for num_components_ in range(1, 9):
        dag2 = CausalDAGalter(data, num_components=num_components_)
        dag2.discover_dag()
        dag2.visualize(graph_type="dag")
        dag2.discover_hierarchical_dag()
        dag2.visualize(graph_type="hierarchical_dag")
        

         
