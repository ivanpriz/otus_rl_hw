import os

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from rest_agent.src.services import AgentService
from rest_agent.src.schemas import GameSetupSchema, StateSchema


app = FastAPI()


@app.post("/initialize")
def initialize(setup_schema: GameSetupSchema):
    app.state.agent_service = AgentService(
        n_actions=setup_schema.n_actions,
        n_dimensions=setup_schema.n_dimensions,
        initial_game_state=setup_schema.game_state,
        save_dir_path=os.path.join(
            os.curdir,
            "dqn_buffer_models",
        )
    )
    app.state.agent_service.build_agent()


# Note: 2 routes below should be called only after /initialize is called
@app.get("/action")
def get_action():
    action = app.state.agent_service.get_action()
    # print(f"Requested action, curr action: {action}")
    return Response(
        content=str(action),
    )


@app.post("/state")
def post_state(state: StateSchema):
    # print("Updating state...")
    app.state.agent_service.update_state(state)
    # print("State updated!")
    return Response(status_code=200)


@app.post("/save")
def save_model():
    app.state.agent_service.save_agent()
    return Response(status_code=200)
