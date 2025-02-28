"""
Gradio web interface implementation for the RAG chatbot.

This module provides a user-friendly chat interface using Gradio framework.
It handles the presentation layer of the application, separating UI concerns
from core business logic.
"""

import logging
import gradio as gr

class GradioInterface:
    def __init__(self, search_service, llm_service):
        """
        Initialize Gradio interface with required services.
        
        Args:
            search_service (SearchService): Service for retrieving relevant context
            llm_service (LLMService): Service for generating answers
        """
        self.search_service = search_service
        self.llm_service = llm_service
        logging.info("GradioInterface initialized with search and LLM services")

    def _chatbot_interface(self, message, history):
        """
        Core chatbot logic that processes user messages and generates responses.
        
        Args:
            message (str): User's question or input
            history (list): List of previous (question, answer) tuples
            
        Returns:
            tuple: (history, history) - Both elements are the same list of message tuples
        """
        try:
            logging.info(f"Processing new message: {message}")
            
            if not message.strip():
                logging.warning("Empty message received")
                history.append((message, "Please enter a valid question."))
                return history, history

            logging.info("Searching for relevant context...")
            search_results = self.search_service.search(message)
            if not search_results:
                logging.warning("No search results found")
                history.append((message, "I couldn't find any relevant information to answer your question."))
                return history, history
            
            logging.info(f"Found {len(search_results)} relevant passages")
            context = "\n\n".join([text for text, score, metadata in search_results])
            
            logging.info("Generating response using LLM...")
            response, _ = self.llm_service.generate_response(message, context)
            if not response:
                logging.warning("LLM returned empty response")
                history.append((message, "I couldn't generate a response. Please try again."))
                return history, history
            
            logging.info("Formatting response with references...")
            formatted_response = self.llm_service.format_response_with_references(
                response, search_results
            )

            logging.info("Response generated successfully")
            history.append((message, formatted_response))
            return history, history

        except Exception as e:
            logging.error(f"Error in chatbot interface: {str(e)}", exc_info=True)
            error_message = f"An error occurred while generating the response: {str(e)}\nSources:"
            history.append((message, error_message))
            return history, history

    def build_interface(self):
        """
        Build and configure the Gradio interface.
        
        Returns:
            gr.Blocks: Configured Gradio interface ready for launch
        """
        with gr.Blocks(css="""
            #user-input {
                padding-top: 2px !important;
                font-size: 16px !important;
                line-height: 1.2 !important;
                background-color: #f9f9f9 !important;
            }
            #submit-button {
                margin-top: 10px !important;
                background-color: #2196F3 !important;
                color: white !important;
            }
        """) as demo:
            gr.Markdown("### RAG Chatbot")
            gr.Markdown("Ask any question about the PDF documents in the knowledge base.")

            chatbot = gr.Chatbot(
                label="Chat History",
                show_label=True,
                height=400
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    lines=2,
                    placeholder="Enter your question about PDF documents...",
                    label="Your Question",
                    elem_id="user-input"
                )
                submit_btn = gr.Button("Send", elem_id="submit-button")
            
            clear_btn = gr.Button("Clear Chat History")
            state = gr.State([])

            submit_btn.click(
                fn=self._chatbot_interface,
                inputs=[msg, state],
                outputs=[chatbot, state]
            )
            msg.submit(
                fn=self._chatbot_interface,
                inputs=[msg, state],
                outputs=[chatbot, state]
            )
            
            clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

        return demo

    def launch(self, server_name="0.0.0.0", server_port=7862, share=False):
        """
        Launch the Gradio interface with specified settings.
        
        Args:
            server_name (str): Host to bind to
            server_port (int): Port to run on
            share (bool): Whether to create a public link
        """
        demo = self.build_interface()
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        ) 