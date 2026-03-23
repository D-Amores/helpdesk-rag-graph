from langgraph.graph import StateGraph, START, END
from graph.state import HelpdeskState
from graph.nodes import RagNode, ClassifyNode, EscalingNode, ResponseNode
from graph.edges import decide_from_clasification, decide_since_human
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


class HelpdeskWorkflow:
    def __init__(self):
        self.graph = None
        self.rag_node = RagNode()
        self.classify_node = ClassifyNode()
        self.escaling_node = EscalingNode()
        self.response_node = ResponseNode()

    def build_graph(self):
        self.graph = StateGraph(HelpdeskState)

        self.graph.add_node("rag", self.rag_node.process_rag)
        self.graph.add_node("classify", self.classify_node.classify_with_context)
        self.graph.add_node("escalated", self.escaling_node.prepare_escaling)
        self.graph.add_node(
            "response_final", self.response_node.generate_final_response
        )
        self.graph.add_node("process_human", self.escaling_node.process_human_response)

        self.graph.add_edge(START, "rag")
        self.graph.add_edge("rag", "classify")

        self.graph.add_conditional_edges(
            "classify",
            decide_from_clasification,
            {
                "response_final": "response_final",
                "escalated": "escalated",
            },
        )

        self.graph.add_conditional_edges(
            "escalated",
            decide_since_human,
            {
                "process_human_response": "process_human",
                "waiting_human": END,  # Pause the graph until the human response
            },
        )

        self.graph.add_edge("process_human", END)
        self.graph.add_edge("response_final", END)

        return self.graph

    def compile_graph(self):
        """Compile the graph with Checkpointer"""

        if not self.graph:
            self.build_graph()

        conn = sqlite3.connect("helpdesk.db", check_same_thread=False)

        checkpointer = SqliteSaver(conn)

        compiled_graph = self.graph.compile(
            checkpointer=checkpointer, interrupt_before=["process_human"]
        )

        return compiled_graph


def create_helpdesk():
    helpdesk = HelpdeskWorkflow()
    return helpdesk.compile_graph()
