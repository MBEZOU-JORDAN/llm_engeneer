import gradio as gr
from deal_agent_framework import DealAgentFramework
from agents.deals import Opportunity, Deal

class App:
    """
    This class encapsulates the Gradio application for "The Price is Right" deal hunting agent.
    It provides a user interface to start the agent framework, view the deals it finds,
    and manually trigger alerts for specific deals.
    """

    def __init__(self):    
        """
        Initializes the App class.
        The agent_framework is initialized to None and will be created when the app starts.
        """
        self.agent_framework = None

    def run(self):
        """
        Launches the Gradio user interface.
        """
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
        
            def table_for(opps):
                """
                Formats a list of Opportunity objects into a list of lists
                suitable for display in a Gradio Dataframe.
                """
                return [[opp.deal.product_description, f"${opp.deal.price:.2f}", f"${opp.estimate:.2f}", f"${opp.discount:.2f}", opp.deal.url] for opp in opps]
        
            def start():
                """
                Initializes the DealAgentFramework and populates the UI
                with any deals found in the memory from previous runs.
                """
                self.agent_framework = DealAgentFramework()
                self.agent_framework.init_agents_as_needed()
                opportunities = self.agent_framework.memory
                table = table_for(opportunities)
                return table
        
            def go():
                """
                Runs the agent framework to find new deals and updates the UI.
                """
                self.agent_framework.run()
                new_opportunities = self.agent_framework.memory
                table = table_for(new_opportunities)
                return table
        
            def do_select(selected_index: gr.SelectData):
                """
                Handles the selection of a row in the opportunities table.
                When a row is selected, it triggers an alert for that opportunity.
                """
                opportunities = self.agent_framework.memory
                row = selected_index.index[0]
                opportunity = opportunities[row]
                self.agent_framework.planner.messenger.alert(opportunity)
        
            # UI Layout
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:24px">"The Price is Right" - Deal Hunting Agentic AI</div>')
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">Autonomous agent framework that finds online deals, collaborating with a proprietary fine-tuned LLM deployed on Modal, and a RAG pipeline with a frontier model and Chroma.</div>')
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">Deals surfaced so far:</div>')
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Description", "Price", "Estimate", "Discount", "URL"],
                    wrap=True,
                    column_widths=[4, 1, 1, 1, 2],
                    row_count=10,
                    col_count=5,
                    max_height=400,
                )
        
            # Load initial data when the UI is loaded.
            ui.load(start, inputs=[], outputs=[opportunities_dataframe])

            # Set up a timer to automatically run the agent framework every 60 seconds.
            timer = gr.Timer(value=60)
            timer.tick(go, inputs=[], outputs=[opportunities_dataframe])

            # Register the selection handler for the dataframe.
            opportunities_dataframe.select(do_select)
        
        # Launch the Gradio app.
        ui.launch(share=False, inbrowser=True)

if __name__=="__main__":
    App().run()
    