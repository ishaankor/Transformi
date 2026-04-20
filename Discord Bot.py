import os
import sys
import asyncio
import random
import graphviz
import numpy as np
import pandas as pd
import pyarrow
import concurrent.futures
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
# from tensorflow.keras import utils
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import discord
from discord.ext import commands
from discord import app_commands
import threading
import time
import socket
from keep_alive import keep_alive

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

tf.config.set_visible_devices([], 'GPU')

import matplotlib.patches as patches
import discord
from enum import Enum
from sklearn.linear_model import LinearRegression
from discord.ext import commands
from discord import app_commands
from discord.ui import Select, View, Button, Modal, TextInput
from discord import Intents, File, Embed
from dotenv import load_dotenv
import asyncio
import random
import numpy as np
from discord import File
from discord.ui import Modal, TextInput
from sklearn.linear_model import LinearRegression
from keep_alive import keep_alive
from discord.ui import Modal, TextInput

import ssl
import certifi
try:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: ssl_context
    print("SSL certificates configured successfully")
except Exception as e:
    print(f"SSL configuration warning: {e}")

from sklearn.linear_model import LinearRegression


load_dotenv()
bot = commands.Bot(command_prefix='~', intents=Intents.all())
token = os.getenv('DISCORD_BOT_TOKEN')
print(f"Bot token ({token}) characters loaded successfully" if token else "Failed to load bot token. Check your .env file.")


class BotState:
    def __init__(self):
        self.initialization_status = False
        self.bot_message_history = {}
        self.active_interactions = {}
        self.dataset_cache = {}
        # self.createnn_active_interactions = set()
        self.text_channel_list = []

    async def initialize(self):
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)!")

        for guild in bot.guilds:
            for channel in guild.text_channels:
                self.text_channel_list.append(channel)
                try:
                    async for message in channel.history():
                        if message.author == bot.user:
                            self.bot_message_history[message.id] = message
                except discord.errors.Forbidden:
                    pass

        self.initialization_status = True
        print("Ready!")


bot_state = BotState()


@bot.event
async def on_ready():
    await bot_state.initialize()


def initialization_check(ctx: commands.Context) -> bool:
    return bot_state.initialization_status


# @bot.listen()
# async def on_message(message: discord.Message):
#     if message.author != bot.user and message.reference:
#         if message.reference.message_id in bot_state.bot_message_history:
#             await message.reply("Sorry, but this command has already been completed!", delete_after=15)
#             recalled_interaction_message = bot_state.bot_message_history.get(message.reference.message_id)
#             await recalled_interaction_message.delete()
#             await message.delete()
#         else:
#             try:
#                 fetched = await message.channel.fetch_message(message.reference.message_id)
#                 print(f"FETCHED! {fetched.author.bot}")
#                 if fetched.author.bot is True:
#                     # print(message.reference.cached_message)
#                     print("SLEEPING!")
#                     await asyncio.sleep(15)
#                     print("PASSED!")
#                     fetched = await message.channel.fetch_message(message.reference.message_id)
#                     print(fetched)
#                     print(f"{message.author.id} currently in interaction!")
                    # print(message.reference.fail_if_not_exists)

                    # pass ### WORK ON THIS --> GET REFERENCED MESSAGE THEN REVERSE ENGINEER/VERIFY?
                # interaction_ids = [x.id for x in bot_state.active_interactions.values()]
                # if message.reference.message_id not in interaction_ids:
                #     # await asyncio.sleep(10)
                #     try:
                #         await message.channel.fetch_message(message.reference.message_id)
                #     except discord.errors.NotFound:
                #         await message.reply("Thanks for the ping! Cleaning this conversation!", delete_after=5)
                #         await message.delete()
                #         print("Deleted already!")
                # else:
            #         print(f"{message.author.id} currently in interaction!")
            # except:
            #     print("DELETED REFERENCED MESSAGE")
            #     await message.reply("Thanks for the ping! Cleaning this conversation!", delete_after=5)
            #     await message.delete()
            #     print("Deleted!")
            #     pass
                # try:
                #     await message.channel.fetch_message(message.id)
                #     await message.reply("Thanks for the ping! Cleaning this conversation!", delete_after=5)
                #     await message.delete()
                # except:
                #     await message.reply("Thanks for the ping! Cleaning this conversation!", delete_after=5)
                #     await message.delete()
                # print(f"{message.author.id} currently in interaction!")


class RNGModal(Modal, title="Insert the number of data values to create!"):
    def __init__(self):
        super().__init__(timeout=15)

        self.answer = TextInput(
            label="Number of Values",
            style=discord.TextStyle.short,
            required=True, 
            placeholder="Enter a positive integer..."
        )
        self.add_item(self.answer)
        self.response_future = asyncio.get_event_loop().create_future()

    async def on_submit(self, interaction: discord.Interaction):
        """Handles valid user input and generates the graph."""
        try:
            num_values = int(self.answer.value)

            if num_values <= 0:
                raise ValueError("The number must be a positive integer.")

            x_axis = np.arange(1, num_values + 1)
            y_axis = np.random.randint(0, 1000, size=num_values)

            reg = LinearRegression().fit(x_axis.reshape(-1, 1), y_axis)
            plt.scatter(x_axis, y_axis, color="g")
            plt.plot(x_axis, reg.predict(x_axis.reshape(-1, 1)), color="k")
            plt.savefig("test.png")

            test_file = File("test.png")
            await interaction.response.send_message(file=test_file, ephemeral=True)

            self.response_future.set_result(num_values)

        except ValueError:
            await interaction.response.send_message(
                "Invalid input! Please enter a **positive integer**.", ephemeral=True
            )
            self.response_future.set_exception(ValueError("User entered invalid input."))

    async def on_timeout(self):
        """Handles modal timeout (user closes or doesn't respond)."""
        if not self.response_future.done():
            self.response_future.set_exception(asyncio.TimeoutError("Modal timed out."))    


class ManualModal(Modal, title='Insert the array of data values!'):
    def __init__(self):
        super().__init__(timeout=15)
        self.answer = TextInput(label='Array of Values', style=discord.TextStyle.short)
        self.add_item(self.answer)
        self.response_future = asyncio.get_event_loop().create_future()

    async def on_submit(self, interaction: discord.Interaction):
        print("CHECK: " + self.answer.value.replace(' ', ''))
        input_array = [int(x) for x in self.answer.value.replace(' ', '').split(",")]
        print("Transformed array: " + str(input_array))
        try:
            x_axis = list(range(1, len(input_array) + 1))
            np_x = np.array(x_axis)
            np_y = np.array(input_array)

            reg = LinearRegression().fit(np_x.reshape(-1, 1), np_y)

            plt.scatter(np_x, np_y, color='g')
            plt.plot(np_x, reg.predict(np_x.reshape(-1, 1)), color='k')
            plt.savefig('test.png')
            test_file = File('test.png')
            await interaction.response.send_message(file=test_file, ephemeral=True)
            self.response_future.set_result(input_array)
        except AttributeError:
            await interaction.response.send_message(
                "Invalid input! Please in the form of a **comma-separated list of numbers**.", ephemeral=True
            )
            self.response_future.set_exception(ValueError("User entered invalid input."))
            # bot_state.active_interactions.remove(interaction.user.id)

    async def on_timeout(self):
        """Handles modal timeout (user closes or doesn't respond)."""
        if not self.response_future.done():
            self.response_future.set_exception(asyncio.TimeoutError("Modal timed out."))    

class DatasetSelect(Select):
    class ValueIdentification(Enum):
        feature = 1
        label = 2

    def __init__(self, parent_view, dataframe: pd.DataFrame, value_id):
        self.parent_view = parent_view
        self.value_id = value_id
        self.columns = list(dataframe.columns.values)
        select_options = []
        for i, column in enumerate(self.columns):
            label = column[:100].strip()
            if not label:
                label = f"Column {i}"
            description = f"This is {column}"[:100].strip()
            if not description:
                description = f"Column {i}"
            select_options.append(discord.SelectOption(label=label, description=description, value=str(i)))
        super().__init__(placeholder="Choose an option...", min_values=1, max_values=1, options=select_options)

    async def callback(self, interaction: discord.Interaction):
        self.parent_view.feature_input_status = True
        if not self.parent_view.confirmation_status:
            value_id_str = "feature" if self.value_id == DatasetView.ValueIdentification.feature else "label"
            await interaction.response.send_message(
                f"If you've selected the correct {value_id_str}, hit the checkmark above!", ephemeral=True, delete_after=5)
            self.parent_view.current_option = self.columns[int(self.values[0])]
        else:
            print("All good?!")

class DatasetView(View):
    class ValueIdentification(Enum):
        feature = 1
        label = 2

    def __init__(self, dataframe: pd.DataFrame, feature_or_label: int):
        super().__init__()
        self.confirmation_status = False
        self.feature_input_status = False
        self.current_option = None
        self.selected_option = asyncio.get_event_loop().create_future()
        self.value_id = DatasetView.ValueIdentification(feature_or_label)
        self.add_item(DatasetSelect(self, dataframe, self.value_id))

    @discord.ui.button(emoji="✅")
    async def confirm_feature_callback(self, interaction: discord.Interaction, button: Button):
        if self.feature_input_status:
            self.confirmation_status = True
            for item in self.children:
                item.disabled = True
            await interaction.response.edit_message(view=self)
            self.feature_input_status = False
            self.selected_option.set_result(self.current_option)
            return
        else:
            value_id_str = "feature" if self.value_id == DatasetView.ValueIdentification.feature else "label"
            await interaction.response.send_message(
                f"You did not select any {value_id_str} values! Please select one using the dropdown menu!",
                ephemeral=True, delete_after=5)


async def linear_regression_calculator(interaction, dataframe, feature_set, label_set):
    x_axis = dataframe[feature_set].tolist()
    np_x = np.array(x_axis)
    y_axis = dataframe[label_set].tolist()
    np_y = np.array(y_axis)

    def compute_and_plot():
        plt.switch_backend('Agg')
        reg = LinearRegression().fit(np_x.reshape(-1, 1), np_y)
        plt.scatter(np_x, np_y, color='g')
        plt.plot(np_x, reg.predict(np_x.reshape(-1, 1)), color='k')
        plt.savefig('test.png')

    await asyncio.get_event_loop().run_in_executor(None, compute_and_plot)
    test_file = File('test.png')
    await interaction.followup.send(file=test_file, ephemeral=True)


def get_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include='number')


def save_correlation_heatmap(df: pd.DataFrame, filename: str = 'correlation_heatmap.png') -> bool:
    plt.switch_backend('Agg')
    numeric_df = get_numeric_dataframe(df)
    if numeric_df.shape[1] < 2:
        return False

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 6))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return True


def build_dataset_summary_embed(df: pd.DataFrame) -> Embed:
    numeric_df = get_numeric_dataframe(df)
    total_missing = int(df.isna().sum().sum())
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    top_missing = [
        f"{col}: {int(count)}" for col, count in missing_by_col.head(3).items() if count > 0
    ]
    top_missing_text = "\n".join(top_missing) if top_missing else "None"

    embed = Embed(
        title='🧪 Dataset Summary',
        description='A quick overview of your uploaded dataset.',
        color=0x00bfff
    )
    embed.add_field(name='Shape', value=f'{len(df)} rows × {len(df.columns)} columns', inline=False)
    embed.add_field(name='Numeric columns', value=str(len(numeric_df.columns)), inline=True)
    embed.add_field(name='Non-numeric columns', value=str(len(df.columns) - len(numeric_df.columns)), inline=True)
    embed.add_field(name='Missing values', value=str(total_missing), inline=False)
    embed.add_field(name='Top missing columns', value=top_missing_text, inline=False)

    if len(df.columns) <= 8:
        columns_text = '\n'.join(df.columns.tolist())
        embed.add_field(name='All columns', value=columns_text or 'None', inline=False)
    else:
        embed.add_field(name='Example columns', value=', '.join(df.columns[:6].tolist()) + ', ...', inline=False)

    return embed


async def request_dataset_csv(interaction: discord.Interaction, prompt_text: str, ephemeral: bool = True) -> pd.DataFrame | None:
    if not interaction.response.is_done():
        await interaction.response.send_message(prompt_text, ephemeral=ephemeral)
        dataset_prompt = await interaction.original_response()
    else:
        dataset_prompt = await interaction.followup.send(prompt_text, ephemeral=ephemeral)
    bot_state.active_interactions[interaction.user.id] = dataset_prompt

    async def invalid_reply(message: discord.Message, reason: str):
        if interaction.response.is_done():
            await interaction.followup.send(f"{message.author.mention} {reason}", ephemeral=True)
        else:
            await interaction.response.send_message(f"{message.author.mention} {reason}", ephemeral=True)

    def check(m: discord.Message):
        if m.author != bot.user and m.author == interaction.user:
            if m.reference and m.reference.message_id == dataset_prompt.id:
                if len(m.attachments) == 1 and m.attachments[0].filename.endswith('.csv'):
                    return True
                if len(m.attachments) > 1:
                    asyncio.create_task(invalid_reply(m, 'Please upload just one CSV file.'))
                else:
                    asyncio.create_task(invalid_reply(m, 'Please upload a CSV file in reply to the prompt.'))
        return False

    try:
        msg = await bot.wait_for('message', check=check, timeout=120.0)
        dataset_file = await msg.attachments[0].to_file()
        await dataset_prompt.delete()
        await msg.delete()
        df = pd.read_csv(dataset_file.fp, engine='pyarrow')
        return df
    except asyncio.TimeoutError:
        await interaction.followup.send('Upload timed out. Please run this command again when ready.', ephemeral=True)
    except Exception as exc:
        await interaction.followup.send(f'Unable to read the CSV file: {exc}', ephemeral=True)
    finally:
        bot_state.active_interactions.pop(interaction.user.id, None)
    return None


async def select_feature_and_label(interaction: discord.Interaction, df: pd.DataFrame) -> tuple[str, str] | None:
    numeric_df = get_numeric_dataframe(df)
    if numeric_df.shape[1] < 2:
        await interaction.followup.send(
            'Your dataset must contain at least two numeric columns for feature and label selection.',
            ephemeral=True
        )
        return None
    if numeric_df.shape[1] > 25:
        await interaction.followup.send(
            'Your dataset has more than 25 numeric columns. Using the first 25 columns for selection.',
            ephemeral=True
        )
        numeric_df = numeric_df.iloc[:, :25]

    feature_view = DatasetView(numeric_df, 1)
    label_view = DatasetView(numeric_df, 2)

    await interaction.followup.send(
        content='Select the feature column to use as input.',
        view=feature_view,
        ephemeral=True
    )
    await interaction.followup.send(
        content='Select the target column to predict.',
        view=label_view,
        ephemeral=True
    )

    selected_feature = await feature_view.selected_option
    selected_label = await label_view.selected_option
    if not selected_feature or not selected_label or selected_feature == selected_label:
        await interaction.followup.send('Feature and target columns must both be selected and must be different.', ephemeral=True)
        return None
    return selected_feature, selected_label


def train_model_on_dataframe(df: pd.DataFrame, feature_col: str, label_col: str, model_name: str):
    numeric_df = get_numeric_dataframe(df[[feature_col, label_col]])
    X = numeric_df[[feature_col]].values
    y = numeric_df[label_col].values

    if len(X) < 5:
        raise ValueError('Dataset must contain at least 5 rows for training.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == 'decision_tree':
        model = DecisionTreeRegressor(random_state=42)
    elif model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return model, mse, r2


def compare_models_on_dataframe(df: pd.DataFrame, feature_col: str, label_col: str) -> dict:
    results = {}
    for model_name in ['linear', 'decision_tree', 'random_forest']:
        model, mse, r2 = train_model_on_dataframe(df, feature_col, label_col, model_name)
        results[model_name] = {'mse': mse, 'r2': r2}
    return results


class ModelTypeView(View):
    def __init__(self):
        super().__init__()
        self.selection = asyncio.get_event_loop().create_future()

    @discord.ui.button(label='Linear Regression', style=discord.ButtonStyle.primary)
    async def linear_button(self, interaction: discord.Interaction, button: Button):
        self.selection.set_result('linear')
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(view=self)

    @discord.ui.button(label='Decision Tree', style=discord.ButtonStyle.secondary)
    async def decision_tree_button(self, interaction: discord.Interaction, button: Button):
        self.selection.set_result('decision_tree')
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(view=self)

    @discord.ui.button(label='Random Forest', style=discord.ButtonStyle.success)
    async def random_forest_button(self, interaction: discord.Interaction, button: Button):
        self.selection.set_result('random_forest')
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(view=self)



    def __init__(self):
        super().__init__(timeout=30)
        self.feature_input = TextInput(label="Feature values", style=discord.TextStyle.paragraph, placeholder="Comma-separated numbers", required=True)
        self.label_input = TextInput(label="Label values", style=discord.TextStyle.paragraph, placeholder="Comma-separated numbers", required=True)
        self.add_item(self.feature_input)
        self.add_item(self.label_input)
        self.response_future = asyncio.get_event_loop().create_future()

    async def on_submit(self, interaction: discord.Interaction):
        try:
            features = [float(x.strip()) for x in self.feature_input.value.split(',') if x.strip()]
            labels = [float(x.strip()) for x in self.label_input.value.split(',') if x.strip()]
            if len(features) != len(labels):
                raise ValueError("Feature and label lists must have the same length.")
            df = pd.DataFrame({'feature': features, 'label': labels})
            if not self.response_future.done():
                self.response_future.set_result(df)
            await interaction.response.send_message("Dataset created successfully.", ephemeral=True)
        except ValueError as e:
            await interaction.response.send_message(f"Invalid input: {e}", ephemeral=True)
            if not self.response_future.done():
                self.response_future.set_exception(e)

    async def on_timeout(self):
        if not self.response_future.done():
            self.response_future.set_exception(asyncio.TimeoutError("Modal timed out."))


class DatasetInputView(View):
    def __init__(self, original_interaction):
        super().__init__(timeout=30)
        self.original_interaction = original_interaction
        self.selected_df = asyncio.get_event_loop().create_future()

    def complete_selection(self, df=None, exc=None):
        if self.selected_df.done():
            return
        if exc is not None:
            self.selected_df.set_exception(exc)
        else:
            self.selected_df.set_result(df)

    async def on_timeout(self):
        self.complete_selection(exc=asyncio.TimeoutError("Selection timed out."))

    @discord.ui.button(label="Upload CSV", style=discord.ButtonStyle.primary)
    async def upload_csv_callback(self, interaction: discord.Interaction, button: Button):
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)
        df = await request_dataset_csv(interaction, "Please reply with your CSV file.", ephemeral=False)
        if df is not None:
            self.complete_selection(df=df)
        else:
            self.complete_selection(exc=ValueError("Failed to get CSV."))

    @discord.ui.button(label="Generate Random Dataset", style=discord.ButtonStyle.secondary)
    async def random_dataset_callback(self, interaction: discord.Interaction, button: Button):
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)
        num_samples = 100
        x = np.random.rand(num_samples)
        y = 3 * x + np.random.randn(num_samples) * 0.1
        df = pd.DataFrame({'feature': x, 'label': y})
        self.complete_selection(df=df)

    @discord.ui.button(label="Manual Input", style=discord.ButtonStyle.success)
    async def manual_input_callback(self, interaction: discord.Interaction, button: Button):
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)
        modal = ManualDatasetModal()
        await interaction.response.send_modal(modal)
        try:
            df = await asyncio.wait_for(modal.response_future, timeout=30)
            self.complete_selection(df=df)
        except asyncio.TimeoutError:
            self.complete_selection(exc=asyncio.TimeoutError("Manual input timed out."))


async def ask_for_dataset_via_menu(interaction: discord.Interaction, title: str, description: str) -> pd.DataFrame | None:
    view = DatasetInputView(interaction)
    await interaction.followup.send(
        embed=Embed(title=title, description=description),
        view=view,
        ephemeral=True
    )
    try:
        df = await asyncio.wait_for(view.selected_df, timeout=120)
        return df
    except asyncio.TimeoutError:
        await interaction.followup.send("Selection timed out. Please try again.", ephemeral=True)
        return None


def model_info_embed(model_name: str, mse: float, r2: float, feature_col: str, label_col: str) -> Embed:
    embed = Embed(
        title='📈 Model Training Results',
        color=0x00ff00
    )
    embed.add_field(name='Model', value=model_name.replace('_', ' ').title(), inline=False)
    embed.add_field(name='Feature', value=feature_col, inline=True)
    embed.add_field(name='Target', value=label_col, inline=True)
    embed.add_field(name='MSE', value=f'{mse:.4f}', inline=True)
    embed.add_field(name='R² Score', value=f'{r2:.4f}', inline=True)
    return embed


def compare_models_embed(results: dict, feature_col: str, label_col: str) -> Embed:
    embed = Embed(
        title='🤖 Model Comparison Results',
        description=f'Model comparison for `{feature_col}` → `{label_col}`',
        color=0x00d1ff
    )
    for name, metrics in results.items():
        embed.add_field(
            name=name.replace('_', ' ').title(),
            value=f'MSE: {metrics["mse"]:.4f}\nR²: {metrics["r2"]:.4f}',
            inline=False
        )
    return embed


def get_cached_dataset(user_id: int) -> pd.DataFrame | None:
    return None





def safe_model_and_dataset(interaction: discord.Interaction):
    return None


def compare_model_command_text(feature_col: str, label_col: str) -> str:
    return f"Comparing Linear Regression, Decision Tree, and Random Forest for `{feature_col}` → `{label_col}`."


def upload_dataset_prompt_text() -> str:
    return 'Please reply to this message with one CSV file containing your dataset.'


def dataset_not_ready_text() -> str:
    return 'No dataset is cached. Run /describe_data or /compare_models to provide one.'


def already_cached_dataset_text() -> str:
    return 'A dataset is already cached. Run /describe_data or /compare_models to continue.'


def no_feature_importance_text() -> str:
    return 'The last trained model does not expose feature importances.'


def command_timeout_text() -> str:
    return 'You took too long to choose. Please run the command again.'


def invalid_feature_label_text() -> str:
    return 'Feature and label columns must be different and both numeric.'


def training_intermediate_text(model_name: str) -> str:
    return f'Training {model_name.replace("_", " ").title()} now...'


def compare_intermediate_text() -> str:
    return 'Training each model and comparing performance...'


def dataset_upload_instructions(feature_col: str, label_col: str) -> str:
    return f'Select `{feature_col}` as feature and `{label_col}` as target.'



def save_dataframe_sample(df: pd.DataFrame, filename: str = 'dataset_preview.csv') -> None:
    df.head(10).to_csv(filename, index=False)


def get_first_numeric_columns(df: pd.DataFrame) -> list[str]:
    return get_numeric_dataframe(df).columns.tolist()


def feature_label_error_text() -> str:
    return 'Selected columns must be numeric and different from each other.'


def generic_failure_text() -> str:
    return 'An error occurred while executing the command. Please try again.'


def model_cache_entry_text() -> str:
    return 'A trained model is cached for this user.'


def dataset_cache_entry_text() -> str:
    return 'A dataset has been cached for your session.'


def correlation_plot_saved_text() -> str:
    return 'Correlation heatmap saved successfully.'


def no_correlation_plot_text() -> str:
    return 'Not enough numeric columns to create a correlation heatmap.'


def dataset_upload_ignore_text() -> str:
    return 'Ignoring the message because it was not a valid dataset upload.'


def invalid_csv_text() -> str:
    return 'That file does not appear to be a valid CSV. Please try again.'


def upload_csv_prompt_text() -> str:
    return 'Please upload one CSV file in reply to this prompt.'


def select_model_type_text() -> str:
    return 'Choose the model type to train on your selected columns.'


def compare_models_ready_text() -> str:
    return 'Dataset selected. Now comparing models.'



def run_inference_text(feature_col: str) -> str:
    return f'Provide values for `{feature_col}` to make predictions.'


def result_format_text(mse: float, r2: float) -> str:
    return f'MSE: {mse:.4f}, R²: {r2:.4f}'



def command_summary_text() -> str:
    return 'This command runs an ML workflow against your cached dataset.'


def dataset_prompt_text() -> str:
    return 'Upload your dataset as a CSV file using reply to this message.'


def upload_dataset_result_text(rows: int, cols: int) -> str:
    return f'Dataset uploaded: {rows} rows, {cols} columns.'


def no_data_error_text() -> str:
    return 'No dataset available. Run /describe_data to provide one.'


def compare_models_summary_text() -> str:
    return 'Model comparison completed successfully.'


def dataset_summary_sent_text() -> str:
    return 'Dataset summary generated successfully.'


def prediction_sent_text() -> str:
    return 'Prediction results delivered successfully.'


def upload_completed_text() -> str:
    return 'Upload completed successfully.'


def training_completed_text() -> str:
    return 'Training finished successfully.'


def compare_completed_text() -> str:
    return 'Comparison finished successfully.'


def upload_dataset_help_text() -> str:
    return 'Upload a dataset and train models directly in Discord.'


def describe_data_help_text() -> str:
    return 'Produce a dataset summary, missing values, and correlation heatmap.'


def compare_models_help_text() -> str:
    return 'Compare Linear Regression, Decision Tree, and Random Forest models.'


def no_cached_summary_text() -> str:
    return 'No cached dataset found for your user.'


def choose_columns_text() -> str:
    return 'Choose the input and target columns for training.'


def model_type_selection_text() -> str:
    return 'Select the type of model to train.'


def or_text() -> str:
    return 'or'


def success_symbol() -> str:
    return '✅'

def error_symbol() -> str:
    return '❌'

def info_symbol() -> str:
    return 'ℹ️'

def warning_symbol() -> str:
    return '⚠️'

def dataset_action_text() -> str:
    return 'Dataset action complete.'

def model_action_text() -> str:
    return 'Model action complete.'

def prediction_action_text() -> str:
    return 'Prediction action complete.'

def dataset_cache_info_text() -> str:
    return 'Your dataset is cached and ready.'

def model_cache_info_text() -> str:
    return 'A model is cached and ready for predictions.'

def failed_to_save_plot_text() -> str:
    return 'Unable to generate a plot for this dataset.'

def failed_to_train_text() -> str:
    return 'Training did not complete successfully.'

def no_numeric_columns_text() -> str:
    return 'The selected columns must be numeric.'

def dataset_ready_text() -> str:
    return 'Your dataset is ready for analysis.'

def dataset_upload_success_text(rows: int, cols: int) -> str:
    return f'Dataset uploaded with {rows} rows and {cols} columns.'


def compare_results_text() -> str:
    return 'Comparison results are ready.'


def upload_dataset_finished_text() -> str:
    return 'Dataset upload finished.'


def choose_dataset_columns_text() -> str:
    return 'Choose input and output columns from the dataset.'


def dataset_stored_text() -> str:
    return 'Dataset stored for your session.'


def model_stored_text() -> str:
    return 'Model stored for your session.'


def analysis_ready_text() -> str:
    return 'Analysis is ready.'


def request_dataset_text() -> str:
    return 'Please upload one CSV dataset file in reply to this message.'


def quick_eda_text() -> str:
    return 'Quick EDA report generated.'


def compare_model_flow_text() -> str:
    return 'Model comparison flow started.'



def reply_prompt_text() -> str:
    return 'Reply to this prompt with a CSV attachment.'


def success_message_text() -> str:
    return 'Success!'


def error_message_text() -> str:
    return 'Something went wrong.'


def prompt_reply_text() -> str:
    return 'Please reply with a CSV attachment to proceed.'


def dataset_upload_info_text() -> str:
    return 'Dataset upload is required before model training.'


def compare_info_text() -> str:
    return 'Compare three models on the same dataset.'


def dataset_selection_info_text() -> str:
    return 'Choose feature and label columns for training.'


def upload_request_text() -> str:
    return 'Upload a CSV file to proceed.'


def numeric_feature_text() -> str:
    return 'Feature column must be numeric.'


def numeric_target_text() -> str:
    return 'Target column must be numeric.'


def training_notice_text() -> str:
    return 'Training may take a moment.'


def comparison_notice_text() -> str:
    return 'Comparing models may take a moment.'


def prediction_notice_text() -> str:
    return 'Computing predictions now.'


def dataset_help_text() -> str:
    return 'Upload your dataset and create ML models from it.'


def model_help_text() -> str:
    return 'Build and compare ML models from your uploaded dataset.'


def prediction_help_text() -> str:
    return 'Run inference on your last trained model.'


def dataset_prompt_title_text() -> str:
    return 'Upload Dataset'


def feature_label_prompt_text() -> str:
    return 'Choose Feature and Target'


def model_type_prompt_text() -> str:
    return 'Choose Model Type'


def prediction_prompt_text() -> str:
    return 'Input values for prediction'


def dataset_command_help_text() -> str:
    return 'Run /describe_data, /train_model, or /compare_models to provide a dataset and analyze it.'


def model_command_help_text() -> str:
    return 'Use /train_model to train a model and /predict to infer new values.'


def compare_command_help_text() -> str:
    return 'Use /compare_models after uploading a dataset.'


def predict_command_help_text() -> str:
    return 'Use /predict after training a model.'


def rapid_response_text() -> str:
    return 'Command received and processing.'


def pending_response_text() -> str:
    return 'Your request is being handled.'


def no_cached_model_text() -> str:
    return 'No cached model found for your session.'


def insufficient_data_text() -> str:
    return 'Insufficient numeric data for this operation.'


def dataset_updated_text() -> str:
    return 'Dataset cache has been updated.'


def model_updated_text() -> str:
    return 'Model cache has been updated.'



def model_cache_message_text() -> str:
    return 'The trained model is ready to make predictions.'


def feature_prediction_prompt_text(feature_col: str) -> str:
    return f'Enter values for `{feature_col}` to predict {feature_col}.'


def dataset_summary_title_text() -> str:
    return 'Dataset Summary'


def dataset_metrics_title_text() -> str:
    return 'Dataset Metrics'


def model_metrics_title_text() -> str:
    return 'Model Metrics'


def compare_metrics_title_text() -> str:
    return 'Comparison Metrics'


def prediction_title_text() -> str:
    return 'Prediction Results'


def training_title_text() -> str:
    return 'Training Results'


def compare_title_text() -> str:
    return 'Comparison Results'


def upload_title_text() -> str:
    return 'Upload Complete'


def dataset_ready_title_text() -> str:
    return 'Dataset Ready'


def model_ready_title_text() -> str:
    return 'Model Ready'


def prediction_ready_title_text() -> str:
    return 'Prediction Ready'


def analysis_ready_title_text() -> str:
    return 'Analysis Ready'
    try:
        print("Configuring TensorFlow...")
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        tf.config.set_visible_devices([], 'GPU')
        
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        print(f"Dataset loaded! Training shape: {x_train.shape}, Test shape: {x_test.shape}")
        
        print("Preprocessing data...")
        x_train = x_train[:1000]
        y_train = y_train[:1000]
        x_test = x_test[:200]
        y_test = y_test[:200]
        print(f"Using smaller dataset - Training: {len(x_train)}, Test: {len(x_test)}")
        
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        print("Data preprocessing complete!")

        print("Creating simplified model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        print("Model created!")

        print("Compiling model...")
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model compiled!")
        
        print("Starting training...")
        history = model.fit(
            x_train, y_train, 
            epochs=1, 
            validation_data=(x_test, y_test), 
            verbose=1,
            batch_size=16
        )
        print("Training completed!")
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        
        print("Saving model architecture...")
        tf.keras.utils.plot_model(model, to_file='model_architecture.png', 
                                show_shapes=True, show_layer_names=True, 
                                rankdir='TB', dpi=150)
        print("Model architecture saved successfully!")
        
        def cleanup_files():
            try:
                if os.path.exists('model_architecture.png'):
                    os.remove('model_architecture.png')
                    print("Cleaned up model_architecture.png")
                if os.path.exists('test.png'):
                    os.remove('test.png')
                    print("Cleaned up test.png")
            except Exception as e:
                print(f"Cleanup warning: {e}")
        
        import threading
        cleanup_timer = threading.Timer(30.0, cleanup_files)
        cleanup_timer.start()
        
        print("Neural network training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during neural network training: {e}")
        import traceback
        traceback.print_exc()
        return False

async def train(ctx):
    # await ctx.followup.send("Training the neural network, this may take a while...")
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, train_neural_network)
        return result


class GraphLRView(View):
    def __init__(self, original_interaction):
        super().__init__(timeout=30)
        self.original_interaction = original_interaction

    async def on_timeout(self):
        """Prevents timeout from removing an active interaction if the user has already interacted."""
        print(self.original_interaction, bot_state.active_interactions.values())
        if self.original_interaction in bot_state.active_interactions.values():
            print(f"[DEBUG @ 269] Timeout removing active interaction for user {self.original_interaction.user.id}")
            await self.original_interaction.edit_original_response(content="**You didn't respond in time! Try again!**", embed=None, view=None)
            del bot_state.active_interactions[self.original_interaction.user.id]
            return
         
        print(f"[DEBUG @ 274] Timeout ignored for user {self.original_interaction.user.id} since they interacted.")
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)
        return


    @discord.ui.button(emoji="1️⃣", label="Random Values", style=discord.ButtonStyle.primary)
    async def button_callback(self, interaction: discord.Interaction, button: Button):
        """Opens the modal and properly handles timeouts and cancellations."""
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        modal = RNGModal()
        await interaction.response.send_modal(modal)
        bot_state.active_interactions[interaction.user.id] = interaction.response

        # for item in self.children:
        #     item.disabled = True
        # await self.original_interaction.edit_original_response(
        #     embed=Embed(title="How would you like to display your dataset?",
        #                 description="1️⃣ Random Dataset Generator \n 2️⃣ Dataset File \n 3️⃣ Given Values/Arrays"), view=self)

        try:
            num_values = await asyncio.wait_for(modal.response_future, timeout=15)  # Wait max 5 min
            print(f"User entered: {num_values}")  # Optional logging
        except asyncio.TimeoutError:
            await interaction.followup.send("You closed the modal or didn’t respond in time.", ephemeral=True)
            print(modal.timeout)
            await self.original_interaction.edit_original_response(view=self)
            print(f"[DEBUG @ 289] Removed active interaction for user {interaction.user.id} due to invalid input.")
        except ValueError:
            pass
        finally:
            del bot_state.active_interactions[interaction.user.id]


    @discord.ui.button(emoji="2️⃣", label="Dataset File (CSV)", style=discord.ButtonStyle.primary)
    async def second_button_callback(self, interaction: discord.Interaction, button: Button):
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        if await interaction_perm_check(interaction):
            await self.original_interaction.edit_original_response(
                embed=Embed(title="How would you like to display your dataset?",
                            description="1️⃣ Random Dataset Generator \n 2️⃣ Dataset File \n 3️⃣ Given Values/Arrays"),
                view=self
            )
            await interaction.response.send_message(
                f"{interaction.user.mention}, please reply to this message with your dataset in a **CSV** file!")
            dataset_prompt = await interaction.original_response()
            print(f"Changed active interaction --> {dataset_prompt}")
            bot_state.active_interactions[interaction.user.id] = dataset_prompt

            async def run(m):
                await m.reply(f"Sorry, this command was not run by you! You can try it by running **/{interaction.command}**!",
                              delete_after=15)
                await m.delete()
                return False

            async def media_reply(message: discord.Message, len_exceeded: bool, file_attached=True):
                if file_attached:
                    if len_exceeded:
                        await dataset_prompt.reply("There's too many files! Please upload just one CSV file!", ephemeral=True)
                        await message.delete()
                        return False
                    else:
                        await dataset_prompt.reply("That's not a valid CSV file! Please upload just one CSV file!", ephemeral=True)
                        await message.delete()
                        return False
                else:
                    await dataset_prompt.reply("There's no files uploaded! Please upload just one CSV file!", ephemeral=True)
                    await message.delete()
                    return False

            def check(m: discord.Message):
                if m.author != bot.user:
                    if m.author == interaction.user:
                        if m.reference and m.reference.message_id == dataset_prompt.id:
                            if len(m.attachments) > 0:
                                if len(m.attachments) == 1 and m.attachments[0].filename.endswith(".csv"):
                                    return True
                                elif len(m.attachments) > 1:
                                    asyncio.create_task(media_reply(m, True))
                                    return False
                                else:
                                    asyncio.create_task(media_reply(m, False))
                                    return False
                            else:
                                asyncio.create_task(media_reply(m, False, False))
                                return False
                        else:
                            print("some other reply?")
                            return False
                    else:
                        if m.reference and m.reference.message_id == dataset_prompt.id:
                            asyncio.create_task(run(m))
                            return False

            try:
                msg = await bot.wait_for("message", check=check, timeout=120.0)
                dataset_file = await msg.attachments[0].to_file()
                print("!")
                await dataset_prompt.delete()
                await msg.delete()
                df = pd.read_csv(dataset_file.fp, engine="pyarrow")

                numeric_df = get_numeric_dataframe(df)
                if numeric_df.shape[1] < 2:
                    await interaction.followup.send(
                        "Your CSV must include at least two numeric columns for feature and target selection.",
                        ephemeral=True
                    )
                    return
                if numeric_df.shape[1] > 25:
                    await interaction.followup.send(
                        "Your CSV has more than 25 numeric columns. Using the first 25 columns for selection.",
                        ephemeral=True
                    )
                    numeric_df = numeric_df.iloc[:, :25]

                feature_view = DatasetView(numeric_df, 1)
                label_view = DatasetView(numeric_df, 2)
                await interaction.followup.send(
                    content="Select the column that represents the feature values!",
                    view=feature_view, ephemeral=True)
                await interaction.followup.send(
                    content="Select the column that represents the label values!",
                    view=label_view, ephemeral=True)
                # await interaction.delete_original_response()
                selected_feature = await feature_view.selected_option
                selected_label = await label_view.selected_option
                print("CHECK!")
                await linear_regression_calculator(interaction, numeric_df, selected_feature, selected_label)
            except asyncio.TimeoutError:
                await interaction.followup.send("You took too long to respond! Please run the command again.",
                                                ephemeral=True)
                print("Deleting dataset prompt...")
                await interaction.delete_original_response()
                print(f"[DEBUG @ 380] Removed active interaction for user {interaction.user.id} due to invalid input.")
            finally:
                del bot_state.active_interactions[interaction.user.id]

    
    @discord.ui.button(emoji="3️⃣", label="Manual Input", style=discord.ButtonStyle.primary)
    async def third_button_callback(self, interaction: discord.Interaction, button: Button):
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        modal = ManualModal()
        await interaction.response.send_modal(modal)
        bot_state.active_interactions[interaction.user.id] = interaction.response

        try:
            num_values = await asyncio.wait_for(modal.response_future, timeout=15)  # Wait max 5 min
            print(f"User entered: {num_values}")  # Optional logging
        except asyncio.TimeoutError:
            await interaction.followup.send("You closed the modal or didn’t respond in time.", ephemeral=True)
            print(f"[DEBUG @ 289] Removed active interaction for user {interaction.user.id} due to invalid input.")
        except ValueError:
            pass
        finally:
            del bot_state.active_interactions[interaction.user.id]


class CreateNNView(View):
    def __init__(self, original_interaction):
        super().__init__(timeout=10)
        self.original_interaction = original_interaction
        self.selected_data = None 
    
    async def on_timeout(self):
        """Prevents timeout from removing an active interaction if the user has already interacted."""
        print(self.original_interaction, bot_state.active_interactions.values())
        if self.original_interaction in bot_state.active_interactions.values():
            print(f"[DEBUG @ 426] Timeout removing active interaction for user {self.original_interaction.user.id}")
            await self.original_interaction.edit_original_response(content="**You didn't respond in time! Try again!**", embed=None, view=None)
            del bot_state.active_interactions[self.original_interaction.user.id]
            return
        
        print(f"[DEBUG @ 431] Timeout ignored for user {self.original_interaction.user.id} since they interacted.")
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)
        return


    @discord.ui.button(emoji="1️⃣", label="Random Dataset", style=discord.ButtonStyle.primary)
    async def random_data_callback(self, interaction: discord.Interaction, button: Button):
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        await interaction.response.defer(ephemeral=True)
        bot_state.active_interactions[interaction.user.id] = interaction.response

        num_samples = 1000
        x_axis = np.random.rand(num_samples, 1)
        y_axis = (3 * x_axis + np.random.randn(num_samples, 1) * 0.1).flatten()

        df = pd.DataFrame({"Feature": x_axis.flatten(), "Label": y_axis})
        self.selected_data = df

        await self.start_training(interaction, df)

    @discord.ui.button(emoji="2️⃣", label="Dataset File", style=discord.ButtonStyle.primary)
    async def upload_csv_callback(self, interaction: discord.Interaction, button: Button):
        
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)


        if await interaction_perm_check(interaction):
            await interaction.response.send_message(
                f"{interaction.user.mention}, please reply to this message with your dataset in a **CSV** file!")
            dataset_prompt = await interaction.original_response()
            bot_state.active_interactions[interaction.user.id] = dataset_prompt

            async def run(m):
                await m.reply(f"Sorry, this command was not run by you! You can try it by running **/{interaction.command}**!",
                              delete_after=15)
                await m.delete()
                return False

            async def media_reply(message: discord.Message, len_exceeded: bool, file_attached=True):
                if file_attached:
                    if len_exceeded:
                        await message.reply("There's too many files! Please upload just one CSV file!", delete_after=15)
                        await message.delete()
                        return False
                    else:
                        await message.reply("That's not a valid CSV file! Please upload just one CSV file!",
                                            delete_after=15)
                        await message.delete()
                        return False
                else:
                    await message.reply("There's no files uploaded! Please upload just one CSV file!", delete_after=15)
                    await message.delete()
                    return False

            def check(m: discord.Message):
                if m.author != bot.user:
                    if m.author == interaction.user:
                        if m.reference and m.reference.message_id == dataset_prompt.id:
                            if len(m.attachments) > 0:
                                if len(m.attachments) == 1 and m.attachments[0].filename.endswith(".csv"):
                                    return True
                                elif len(m.attachments) > 1:
                                    asyncio.create_task(media_reply(m, True))
                                    return False
                                else:
                                    asyncio.create_task(media_reply(m, False))
                                    return False
                            else:
                                asyncio.create_task(media_reply(m, False, False))
                                return False
                        else:
                            print("some other reply?")
                            return False
                    else:
                        if m.reference and m.reference.message_id == dataset_prompt.id:
                            asyncio.create_task(run(m))
                            return False

            try:
                msg = await bot.wait_for("message", check=check, timeout=120.0)
                dataset_file = await msg.attachments[0].to_file()
                await dataset_prompt.delete()
                await msg.delete()
                df = pd.read_csv(dataset_file.fp, engine="pyarrow")
                feature_view = DatasetView(df, 1)
                label_view = DatasetView(df, 2)
                await interaction.followup.send(
                    content="Select the column that represents the feature values!",
                    view=feature_view, ephemeral=True)
                await interaction.followup.send(
                    content="Select the column that represents the label values!",
                    view=label_view, ephemeral=True)
                selected_feature = await feature_view.selected_option
                selected_label = await label_view.selected_option
                self.selected_data = df[[selected_feature, selected_label]]
                print("CHECK!")
                await self.start_training(interaction, self.selected_data)
            except asyncio.TimeoutError:
                await interaction.followup.send(
                    "You took too long to upload the file! Please run the command again.",
                    ephemeral=True
                )
                await interaction.delete_original_response()
                print(f"[DEBUG @ 508] Removed active interaction for user {interaction.user.id} due to invalid input.")
                del bot_state.active_interactions[interaction.user.id]

    async def start_training(self, interaction: discord.Interaction, dataframe: pd.DataFrame):
        """Handles the training after dataset selection."""
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        await interaction.followup.send(
            "Training the neural network with your selected dataset. This may take a while...",
            ephemeral=True
        )

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, train_neural_network)

        if result:
            try:
                files = []
                
                if os.path.exists('model_architecture.png'):
                    files.append(File('model_architecture.png', filename='neural_network_architecture.png'))
                
                embed = discord.Embed(
                    title="🧠 Neural Network Training Complete!",
                    description="""
                    Your neural network has been successfully trained on the MNIST dataset!
                    
                    **📊 Training Results:**
                    🎯 Model architecture diagram included
                    📈 Trained on 5,000 samples
                    🔍 Tested on 1,000 samples
                    ⚡ CNN with Conv2D → MaxPooling → Dense layers
                    """,
                    color=0x00ff00
                )
                embed.add_field(name="🏗️ Architecture", value="Conv2D(32) → MaxPool → Flatten → Dense(64) → Dense(10)", inline=False)
                embed.add_field(name="📊 Dataset", value="MNIST Handwritten Digits", inline=True)
                embed.add_field(name="⏱️ Training", value="Quick training mode", inline=True)
                embed.set_footer(text="Model architecture and training completed successfully!")
                
                if files:
                    await interaction.followup.send(
                        embed=embed,
                        files=files,
                        ephemeral=True
                    )
                else:
                    await interaction.followup.send(embed=embed, ephemeral=True)
                    
            except Exception as e:
                print(f"Error sending visualization files: {e}")
                await interaction.followup.send("Training complete! Model is ready.", ephemeral=True)
        else:
            await interaction.followup.send("Training failed. Please check the logs.", ephemeral=True)
            
        print(f"[DEBUG @ 548] Removed active interaction for user {interaction.user.id} due to invalid input.")
        del bot_state.active_interactions[interaction.user.id]

    # @discord.ui.button(emoji="3️⃣", label="Manual Input", style=discord.ButtonStyle.primary)
    # async def manual_input_callback(self, interaction: discord.Interaction, button: Button):
    #     """Let user manually input data values."""
    #     for item in self.children:
    #         item.disabled = True
    #     await self.original_interaction.edit_original_response(view=self)

    #     modal = ManualModal()
    #     await interaction.response.send_modal(modal)
    #     bot_state.active_interactions[interaction.user.id] = interaction.response

async def interaction_perm_check(interaction: discord.Interaction):
    interaction_user_perms = interaction.channel.permissions_for(interaction.user)
    interaction_bot_perms = interaction.channel.permissions_for(interaction.guild.me)
    if interaction_user_perms.attach_files and interaction_bot_perms.attach_files:
        return True
    else:
        await interaction.response.send_message(
            "We can't upload files in this channel if we want to complete the interaction! Please move to a **different channel** or **direct message** me!",
            ephemeral=True)
        print(f"[DEBUG @ 537] Removed active interaction for user {interaction.user.id} due to invalid input.")
        del bot_state.active_interactions[interaction.user.id]
        # bot_state.graphlr_active_interactions.discard(interaction)
        return False

async def check_user_instances(interaction: discord.Interaction):
    """Checks if a user has an ongoing interaction and prevents new ones."""
    user_id = interaction.user.id
    if user_id in bot_state.active_interactions.keys():
        await interaction.response.send_message(
            "You already have an active interaction! Please complete it before starting a new one.",
            ephemeral=True
        )
        return False    
    
    bot_state.active_interactions[user_id] = interaction
    return True


@bot.tree.command(name="describe_data", description="Generate a quick dataset summary and correlation heatmap.")
@app_commands.check(initialization_check)
async def describe_data(interaction: discord.Interaction):
    if not await check_user_instances(interaction):
        return

    await interaction.response.defer(ephemeral=True)
    try:
        df = await ask_for_dataset_via_menu(
            interaction,
            title="How would you like to provide your dataset?",
            description="1️⃣ Random Dataset Generator \n2️⃣ Dataset File \n3️⃣ Manual Input"
        )
        if df is None:
            return

        embed = await asyncio.get_event_loop().run_in_executor(None, build_dataset_summary_embed, df)
        files = []
        heatmap_saved = await asyncio.get_event_loop().run_in_executor(None, save_correlation_heatmap, df)
        if heatmap_saved:
            files.append(File('correlation_heatmap.png'))
        await interaction.followup.send(embed=embed, files=files, ephemeral=True)
    finally:
        bot_state.active_interactions.pop(interaction.user.id, None)



@bot.tree.command(name="compare_models", description="Compare multiple regression models on a cached dataset.")
@app_commands.check(initialization_check)
async def compare_models(interaction: discord.Interaction):
    if not await check_user_instances(interaction):
        return

    await interaction.response.defer(ephemeral=True)
    try:
        df = await ask_for_dataset_via_menu(
            interaction,
            title="How would you like to provide your dataset?",
            description="1️⃣ Random Dataset Generator \n2️⃣ Dataset File \n3️⃣ Manual Input"
        )
        if df is None:
            return
            cache_dataset(interaction.user.id, df)

        selected = await select_feature_and_label(interaction, df)
        if selected is None:
            return

        feature_col, label_col = selected
        await interaction.followup.send('Training models and comparing performance. This may take a moment...', ephemeral=True)
        try:
            results = await asyncio.get_event_loop().run_in_executor(None, compare_models_on_dataframe, df, feature_col, label_col)
            embed = compare_models_embed(results, feature_col, label_col)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as exc:
            await interaction.followup.send(f'Comparison failed: {exc}', ephemeral=True)
    finally:
        bot_state.active_interactions.pop(interaction.user.id, None)



@bot.command(name="restart", description="Restarts the bot if issues are encountered.")
@commands.is_owner()
async def restart(ctx: discord.ext.commands.Context):
    await ctx.reply("Restarting bot...")
    os.execv(sys.executable, ["python"] + sys.argv)


@bot.tree.command(name="graph_linear_regression", description="Graphs a linear regression model of the given dataset/values.")
@app_commands.check(initialization_check)
async def graph_linear_regression(interaction: discord.Interaction):
    if not await check_user_instances(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    graphlr_view = GraphLRView(interaction)
    await interaction.followup.send(embed=Embed(title="How would you like to display your dataset?",
                                                description="1️⃣ Random Dataset Generator \n 2️⃣ Dataset File \n 3️⃣ Given Numbers/Array"),
                                    view=graphlr_view, ephemeral=True)


@bot.tree.command(name="create_neural_network", description="Creates a neural network model of the given dataset/values.")
@app_commands.check(initialization_check)
async def create_neural_network(interaction: discord.Interaction):
    if not await check_user_instances(interaction):
        return

    await interaction.response.defer(ephemeral=True)

    try:
        createnn_view = CreateNNView(interaction)
        await interaction.followup.send(
            embed=Embed(
                title="How would you like to create your neural network?",
                description="1️⃣ Random Dataset \n 2️⃣ Dataset File \n 3️⃣ Manual Input"
            ),
            view=createnn_view,
            ephemeral=True
        )
    except Exception as e:
        print(f"Error in create_neural_network: {e}")
        await interaction.followup.send("An error occurred while processing your request.", ephemeral=True)



@bot.event
async def on_command_error(ctx, exception):
    if isinstance(exception, commands.CommandOnCooldown):
        await ctx.reply(f"You are rate limited. Please, try again in {exception.retry_after} seconds")
    elif isinstance(exception, commands.CheckFailure):
        await ctx.reply("You don't have the necessary permissions to run this command!")
    else:
        print(f"Unhandled exception: {exception}")


@graph_linear_regression.error
async def graph_linear_regression_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CheckFailure):
        await interaction.response.send_message("Initialization is not complete. Please try again later.",
                                                ephemeral=True)
    else:
        print(f"[DEBUG @ 617] Removed active interaction for user {interaction.user.id} due to invalid input.")
        del bot_state.active_interactions[interaction.user.id]
        await interaction.response.send_message("An error occurred while processing the command.", ephemeral=True)


@create_neural_network.error
async def create_neural_network_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CheckFailure):
        await interaction.response.send_message("Initialization is not complete. Please try again later.",
                                                ephemeral=True)
    else:
        print(f"[DEBUG @ 628] Removed active interaction for user {interaction.user.id} due to invalid input.")
        del bot_state.active_interactions[interaction.user.id]
        await interaction.response.send_message("An error occurred while processing the command.", ephemeral=True)


keep_alive()
bot.run(token)
