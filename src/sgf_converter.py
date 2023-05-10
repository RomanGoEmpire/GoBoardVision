import datetime
from datetime import datetime


class SGFConverter:

    def __init__(self, game_history):
        self.game_history = game_history
        self.sgf_moves = 'abcdefghijklmnopqrs'

    def convert_history_to_sgf(self):
        date = self.get_current_date()
        next_player = 'B' if self.game_history[-1][0] == 'W' else 'W'
        text = f"(;GM[1]FF[4]CA[UTF-8]AP[GoBoardVision]KM[6.5]SZ[19]DT[{date}]PL[{next_player}];"
        game = []
        for move in self.game_history:
            game.append(self.convert_move(move))
        return text + ";".join(game)

    def get_current_date(self):
        return datetime.now().strftime('%Y-%m-%d')

    def convert_move(self, move):
        converted_move = f"{move[0]}["
        x = self.sgf_moves[move[1][0]]
        y = self.sgf_moves[move[1][1]]
        converted_move += f"{y}{x}]"
        return converted_move

    def create_file(self, filename):
        with open(filename, 'w') as file:
            file.write(self.convert_history_to_sgf())
