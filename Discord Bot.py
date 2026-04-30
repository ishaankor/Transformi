import os
import glob
import tempfile
import io
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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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
import matplotlib.patches as patches
import discord
from enum import Enum
from sklearn.linear_model import LinearRegression
from discord.ui import Select, View, Button, Modal, TextInput
from discord import Intents, File, Embed
from dotenv import load_dotenv
from keep_alive import keep_alive
import ssl
import certifi
import threading
import time
import socket
from keep_alive import keep_alive

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

tf.config.set_visible_devices([], 'GPU')


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
        self.nn_model_cache = {}
        # self.createnn_active_interactions = set()
        self.text_channel_list = []
        self.user_locks = {} # Add this to handle concurrent command attempts

    def get_user_lock(self, user_id):
        if user_id not in self.user_locks:
            self.user_locks[user_id] = asyncio.Lock()
        return self.user_locks[user_id]

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

            os.remove("test.png") if os.path.exists("test.png") else None

            self.response_future.set_result(num_values)

        except ValueError:
            # await interaction.response.send_message(
            #     "Invalid input! Please enter a **positive integer**.", ephemeral=True
            # )
            print("User entered invalid input for RNGModal.")
            await interaction.response.edit_message(
                embed=Embed(
                    title="⚠️ Invalid input! ⚠️",
                    description="**Please enter a positive integer.**",
                    color=0xff0000,
                ),
                view=None
            )
            # self.complete_selection(exc=asyncio.TimeoutError("Selection timed out."))
            if not self.response_future.done():
                self.response_future.set_exception(ValueError("User entered invalid input."))
            cleanup_interaction(interaction.user.id)

    async def on_timeout(self):
        """Handles modal timeout (user closes or doesn't respond)."""
        # cleanup_interaction(self.original_interaction.user.id)
        # await in.edit_original_response(
        #     embed=Embed(
        #         title="⏱️ Selection Timed Out",
        #         description="**You didn't respond in time!** Please run the command again.",
        #         color=0xff0000
        #     ),
        #     view=None
        # )
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
        try:
            input_array = [int(x) for x in self.answer.value.replace(' ', '').split(",")]
            print("Transformed array: " + str(input_array))
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
        except ValueError:
            print("Invalid input received in ManualModal.")
            await interaction.response.edit_message(
                embed=Embed(
                    title="⚠️ Invalid input! ⚠️",
                    description="**Please enter a positive integer.**",
                    color=0xff0000,
                ),
                view=None
            )
            cleanup_interaction(interaction.user.id)
            if not self.response_future.done():
                print("Setting exception on response future due to invalid input.")
                self.response_future.set_exception(e)
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
        super().__init__(timeout=30)
        self.confirmation_status = False
        self.feature_input_status = False
        self.current_option = None
        self.selected_option = asyncio.get_event_loop().create_future()
        self.value_id = DatasetView.ValueIdentification(feature_or_label)
        self.add_item(DatasetSelect(self, dataframe, self.value_id))

    async def on_timeout(self):
        """If the view times out, set an exception on the future so it doesn't hang."""
        print("DatasetView timed out without selection.")
        if not self.selected_option.done():
            self.selected_option.set_exception(asyncio.TimeoutError("DatasetView timed out."))

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

    async def invalid_reply(message: discord.Message, reason: str):
            if interaction.response.is_done():
                invalid_reply_msg = await interaction.followup.send(f"{message.author.mention}, {reason}", ephemeral=True)
                await invalid_reply_msg.delete(delay=5)
                try:
                    await message.delete()
                except discord.errors.Forbidden as e:
                    print("Failed to delete message due to lack of permissions or in DMs, skipping cleanup.")
                    # await interaction.followup.send(f"{message.author.mention}, I can't delete messages!", ephemeral=True)
                    pass
            else:
                print("Sending initial interaction response for invalid reply.")
                await interaction.response.send_message(f"{message.author.mention} {reason}", ephemeral=True, delete_after=5)
                # await invalid_reply_msg.delete(delay=5)
                try:
                    await message.delete()
                except discord.errors.Forbidden as e:
                    print("Failed to delete message due to lack of permissions or in DMs, skipping cleanup.")
                    # await interaction.followup.send(f"{message.author.mention}, I can't delete messages!", ephemeral=True)
                    pass

    def check(m: discord.Message):
        if m.author != bot.user and m.author == interaction.user:
            if m.reference and m.reference.message_id == dataset_prompt.id:
                if len(m.attachments) == 1 and m.attachments[0].filename.endswith('.csv'):
                    return True
                if len(m.attachments) > 1:
                    asyncio.create_task(invalid_reply(m, 'Please upload just one CSV file.'))
                else:
                    asyncio.create_task(invalid_reply(m, 'Please upload a CSV file in reply to the prompt.'))
                    # await message.delete()
        return False

    try:
        msg = await bot.wait_for('message', check=check, timeout=30.0)
        dataset_file = await msg.attachments[0].to_file()
        await dataset_prompt.delete()
        try:
            await msg.delete()
        except Exception as e:
            print(f"POST-UPLOAD: Failed to delete message due to lack of permissions or in DMs, skipping cleanup. Error: {e}")
            # print("Failed to delete message due to lack of permissions or in DMs, skipping cleanup.")
            # await interaction.followup.send(f"{msg.author.mention}, I can't delete messages!", ephemeral=True)
            pass
        df = pd.read_csv(dataset_file.fp, engine='pyarrow')
        return df
    except asyncio.TimeoutError:
        await interaction.edit_original_response(
            embed=Embed(
                title="⏱️ Upload Timed Out ⏱️",
                description="**You didn't respond in time!** Please run the command again.",
                color=0xff0000
            ),
            view=None
        )
        await asyncio.create_task(safe_delete_message(dataset_prompt))
        # await interaction.followup.send('Upload timed out. Please run this command again when ready.', ephemeral=True)
        # cleanup_interaction(interaction.user.id)
    except Exception as exc:
        await interaction.edit_original_response(
            embed=Embed(
                title="⚠️ Invalid input! ⚠️",
                description="**You put a non-CSV file!** Please run the command again.",
                color=0xff0000
            ),
            view=None
        )
        await asyncio.create_task(safe_delete_message(dataset_prompt))
        # cleanup_interaction(interaction.user.id)
        # await interaction.followup.send(f'Unable to read the CSV file: {exc}', ephemeral=True)
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
        max_columns_msg = await interaction.followup.send(
            'Your dataset has more than 25 numeric columns. Using the first 25 columns for selection.',
            ephemeral=True
        )
        numeric_df = numeric_df.iloc[:, :25]

    feature_view = DatasetView(numeric_df, 1)
    label_view = DatasetView(numeric_df, 2)

    print("Sending feature and label selection views...")

    feature_msg = await interaction.followup.send(
        content='Select the feature column to use as input.',
        view=feature_view,
        ephemeral=True
    )
    label_msg = await interaction.followup.send(
        content='Select the target column to predict.',
        view=label_view,
        ephemeral=True
    )

    try:
        selected_feature, selected_label = await asyncio.gather(
            feature_view.selected_option,
            label_view.selected_option
        )

        print(f"Deleting feature/label selection messages...")  # Debug log before deletion

    except asyncio.TimeoutError:
        print("User did not select feature/label in time.")
        
        await feature_msg.delete()
        await label_msg.delete()

        try:
            await max_columns_msg.delete()
        except Exception as e:
            print(f"max_columns_msg not present: {e}")
        
        await interaction.edit_original_response(
                embed=Embed(
                    title="⏱️ Timed Out! ⏱️",
                    description="**You didn't select a feature and label in time!** Please run the command again.",
                    color=0xff0000,
                ),
                view=None
        )
        cleanup_interaction(interaction.user.id)
        # await interaction.followup.send('Feature and label selection timed out. Please run the command again.', ephemeral=True)
        return None

    if selected_feature == selected_label:
        await interaction.followup.send('Feature and target columns must be different.', ephemeral=True)
        return None
    
    try:
        await max_columns_msg.delete()
    except Exception as e:
        print(f"Failed to delete max_columns_msg: {e}")

    await feature_msg.delete()
    await label_msg.delete()
        
    return selected_feature, selected_label


def train_model_on_dataframe(df: pd.DataFrame, feature_col: str, label_col: str, model_name: str):
    numeric_df = get_numeric_dataframe(df[[feature_col, label_col]])
    X = numeric_df[[feature_col]].values
    y = numeric_df[label_col].values

    if len(X) < 5:
        print(f"Dataset has only {len(X)} rows after filtering for numeric columns. Minimum 5 rows required.")
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
        try:
            model, mse, r2 = train_model_on_dataframe(df, feature_col, label_col, model_name)
            results[model_name] = {'mse': mse, 'r2': r2}
        except ValueError as e:
            print(f"Error occurred while training {model_name} model: {e}")
            raise ValueError(f"Error with {model_name} model: {e}")
    return results

class ManualDatasetModal(Modal, title="Manual Dataset Input"):
    def __init__(self):
        super().__init__(timeout=30)
        
        self.feature_input = TextInput(
            label="Feature Values (X)", 
            style=discord.TextStyle.paragraph, 
            placeholder="e.g., 1, 2, 3, 4, 5", 
            required=True
        )
        
        self.label_input = TextInput(
            label="Label Values (Y)", 
            style=discord.TextStyle.paragraph, 
            placeholder="e.g., 10, 20, 30, 40, 50", 
            required=True
        )
        
        self.add_item(self.feature_input)
        self.add_item(self.label_input)
        
        self.response_future = asyncio.get_event_loop().create_future()

    async def on_submit(self, interaction: discord.Interaction):
        # NO self.stop() here! Modals don't support it, and it was causing the crash.
        print("Received manual dataset input, processing...")
        try:
            features = [float(x.strip()) for x in self.feature_input.value.split(',') if x.strip()]
            labels = [float(x.strip()) for x in self.label_input.value.split(',') if x.strip()]
            
            if len(features) != len(labels):
                raise ValueError(f"Length mismatch: {len(features)} features vs {len(labels)} labels.")
            
            if len(features) < 2:
                raise ValueError("Please enter at least 2 data points to create a dataset.")

            df = pd.DataFrame({'feature': features, 'label': labels})
            
            # If successful, just defer to close the modal smoothly
            await interaction.response.defer()
            
            if not self.response_future.done():
                self.response_future.set_result(df)

        except ValueError as e:
            await interaction.response.edit_message(
                embed=Embed(
                    title="⚠️ Invalid Input! ⚠️",
                    description=f"**Please make sure you are only entering comma-separated numbers!**",
                    color=0xff0000
                ),
                view=None
            )
            cleanup_interaction(interaction.user.id)

            print(f"Error processing manual dataset input: {e}")
            
            if not self.response_future.done():
                print("Setting exception on response future due to invalid input.")
                self.response_future.set_exception(e)
            
    async def on_timeout(self):
        if not self.response_future.done():
            self.response_future.set_exception(asyncio.TimeoutError("Modal timed out."))

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
            # await interaction.response.send_message("Dataset created successfully.", ephemeral=True, delete_after=5)
        except ValueError as e:
            # await interaction.response.send_message(f"Invalid input: {e}", ephemeral=True)
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
        await self.original_interaction.edit_original_response(
            embed=Embed(
                title="⏱️ Selection Timed Out ⏱️",
                description="**You didn't respond in time!** Please run the command again.",
                color=0xff0000
            ),
            view=None
        )
        self.complete_selection(exc=asyncio.TimeoutError("Selection timed out."))
        cleanup_interaction(self.original_interaction.user.id)

    @discord.ui.button(label="Upload CSV", style=discord.ButtonStyle.primary)
    async def upload_csv_callback(self, interaction: discord.Interaction, button: Button):
        self.stop()
        # if not interaction.permissions.attach_files:
        #     print("User lacks attach files permission, denying CSV upload.")
        #     await self.original_interaction.edit_original_response(
        #         embed=Embed(
        #             title="⚠️ Permission Error! ⚠️",
        #             description="**You lack the 'Attach Files' permission to do this.** Please run the command again.",
        #             color=0xff0000
        #         ),
        #         view=None
        #     )
        #     cleanup_interaction(interaction.user.id)
        #     # self.complete_selection(exc=PermissionError("User lacks Manage Messages permission."))
        #     return
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)
        if await interaction_perm_check(interaction):
            df = await request_dataset_csv(interaction, "Please reply with your CSV file.", ephemeral=False)
            if df is not None:
                self.complete_selection(df=df)
            else:
                await self.original_interaction.edit_original_response(
                    embed=Embed(
                        title="⏱️ Upload Timed Out! ⏱️",
                        description="**You didn't respond in time!** Please run the command again.",
                        color=0xff0000
                    ),
                    view=None
                )
                cleanup_interaction(self.original_interaction.user.id)
                self.complete_selection(df=None)

    @discord.ui.button(label="Generate Random Dataset", style=discord.ButtonStyle.secondary)
    async def random_dataset_callback(self, interaction: discord.Interaction, button: Button):
        self.stop()
        for item in self.children:
            item.disabled = True
        print("Generating random dataset...")
        await interaction.response.edit_message(view=self)
        # bot_state.active_interactions[interaction.user.id] = interaction.response
        num_samples = 100
        x = np.random.rand(num_samples)
        y = 3 * x + np.random.randn(num_samples) * 0.1
        df = pd.DataFrame({'feature': x, 'label': y})
        print("Random dataset generated.")
        self.complete_selection(df=df)
        print("Random dataset selection complete.")
        # cleanup_interaction(interaction.user.id)
        return 

    @discord.ui.button(label="Manual Input", style=discord.ButtonStyle.success)
    async def manual_input_callback(self, interaction: discord.Interaction, button: Button):
        self.stop()
        
        modal = ManualDatasetModal()
        await interaction.response.send_modal(modal)
        
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)
    
        try:
            df = await asyncio.wait_for(modal.response_future, timeout=30)
            print("Manual dataset successfully captured.")
            
            self.complete_selection(df=df)
            
        except asyncio.TimeoutError:
            print("Manual input modal timed out.")
            cleanup_interaction(interaction.user.id)
            self.complete_selection(exc=asyncio.TimeoutError("Manual input timed out."))
            
        except ValueError:
            print("User entered invalid manual data.")
            cleanup_interaction(interaction.user.id)
            self.complete_selection(exc=ValueError("Invalid manual data."))


async def ask_for_dataset_via_menu(interaction: discord.Interaction, title: str, description: str) -> pd.DataFrame | None:
    view = DatasetInputView(interaction)
    await interaction.followup.send(
        embed=Embed(title=title, description=description),
        view=view,
        ephemeral=True
    )
    df = 0
    try:
        df = await asyncio.wait_for(view.selected_df, timeout=30)
        # print(df)
        return df
    except asyncio.TimeoutError:
        print("Dataset selection timed out.")
        # print(df)
        await interaction.edit_original_response(
            embed=Embed(
                title="⏱️ Timed Out! ⏱️",
                description="**You didn't respond in time!** Please run the command again.",
                color=0xff0000
            ),
            view=None
        )
        return None
    except ValueError as e:
        print(f"Error processing dataset input: {e}")
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

def train_neural_network(df, feature_cols, label_col, model_params=None):
    if model_params is None:
        model_params = {'epochs': 5, 'hidden_units': 32, 'task': 'regression'}
    
    try:
        print("Configuring TensorFlow...")
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.set_visible_devices([], 'GPU')
        
        print("Preprocessing user dataset...")
        X = df[feature_cols].values.astype('float32')
        y = df[label_col].values
        
        # Determine task
        if pd.api.types.is_numeric_dtype(df[label_col]):
            task = 'regression'
            y = y.astype('float32')
            num_classes = 1
        else:
            task = 'classification'
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y).astype('int32')
            num_classes = len(np.unique(y))
        
        # Normalize X
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Dataset split - Training: {len(X_train)}, Test: {len(X_test)}")
        
        print("Creating model...")
        input_shape = (X_train.shape[1],)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(model_params['hidden_units'], activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(num_classes, activation='softmax' if model_params['task'] == 'classification' else None)
        ])
        print("Model created!")
        
        print("Compiling model...")
        if model_params['task'] == 'classification':
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("Model compiled!")
        
        print("Starting training...")
        history = model.fit(
            X_train, y_train, 
            epochs=model_params['epochs'], 
            validation_data=(X_test, y_test), 
            verbose=1,
            batch_size=16
        )
        print("Training completed!")
        
        if model_params['task'] == 'classification':
            train_metric = history.history['accuracy'][-1]
            val_metric = history.history['val_accuracy'][-1]
            print(f"Training accuracy: {train_metric:.4f}, Validation accuracy: {val_metric:.4f}")
        else:
            train_metric = history.history['mae'][-1]
            val_metric = history.history['val_mae'][-1]
            print(f"Training MAE: {train_metric:.4f}, Validation MAE: {val_metric:.4f}")
        
        print("Saving model architecture...")
        try:
            tf.keras.utils.plot_model(model, to_file='model_architecture.png', 
                                    show_shapes=True, show_layer_names=True, 
                                    rankdir='TB', dpi=150)
            print("Model architecture saved successfully!")
        except ImportError as e:
            print(f"Skipping architecture plot (Graphviz not installed): {e}")
        except Exception as e:
            print(f"Failed to save architecture plot: {e}")
        
        def cleanup_files():
            try:
                if os.path.exists('model_architecture.png'):
                    os.remove('model_architecture.png')
                    print("Cleaned up model_architecture.png")
                if os.path.exists('test.png'):
                    os.remove('test.png')
                    print("Cleaned up test.png")
                if os.path.exists('regression.png'):
                    os.remove('regression.png')
                    print("Cleaned up regression.png")
            except Exception as e:
                print(f"Cleanup warning: {e}")
        
        import threading
        cleanup_timer = threading.Timer(30.0, cleanup_files)
        cleanup_timer.start()
        
        print("Neural network training completed successfully!")
        return model, history, scaler
        
    except Exception as e:
        print(f"Error during neural network training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

async def train(ctx):
    # await ctx.followup.send("Training the neural network, this may take a while...")
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, train_neural_network)
        return result

async def calculate_linear_regression(interaction: discord.Interaction, df: pd.DataFrame, feature_col: str, label_col: str):
    """Calculates linear regression, plots it in memory, and sends it to Discord."""
    
    await interaction.followup.send(
        f"Calculating linear regression for **{feature_col}** ➡️ **{label_col}**...", 
        ephemeral=True
    )

    def compute_and_plot():        
        plt.switch_backend('Agg')
        
        X = df[feature_col].values.reshape(-1, 1)
        y = df[label_col].values
        
        reg = LinearRegression().fit(X, y)
        predictions = reg.predict(X)
        r2_score = reg.score(X, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color="g", alpha=0.5, label="Data Points")
        plt.plot(X, predictions, color="k", linewidth=2, label=f"Best Fit Line (R² = {r2_score:.4f})")
        
        plt.xlabel(feature_col)
        plt.ylabel(label_col)
        plt.title(f"Linear Regression: {feature_col} vs {label_col}")
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0) 
        plt.close()
        
        return buf, r2_score, slope, intercept

    try:
        loop = asyncio.get_running_loop()
        buf, r2, slope, intercept = await loop.run_in_executor(None, compute_and_plot)

        plot_file = File(fp=buf, filename="regression.png")
        
        embed = Embed(
            title="📈 Linear Regression Results",
            description=f"**Equation:** `y = {slope:.4f}x + {intercept:.4f}`\n**R² Score:** `{r2:.4f}`",
            color=0x00ff00
        )
        embed.set_image(url="attachment://regression.png")

        await interaction.followup.send(embed=embed, file=plot_file, ephemeral=True)

        os.remove("regression.png") if os.path.exists("regression.png") else None
        
    except Exception as e:
        await interaction.followup.send(f"An error occurred during calculation: {e}", ephemeral=True)
    
class GraphLRView(View):
    def __init__(self, original_interaction):
        super().__init__(timeout=30)
        self.original_interaction = original_interaction

    async def on_timeout(self):
        """Prevents timeout from removing an active interaction if the user has already interacted."""
        debug_values = []
        for v in bot_state.active_interactions.values():
            if isinstance(v, tuple):
                debug_values.append(v[0])
            else:
                debug_values.append(v)
        print(self.original_interaction, debug_values)
        
        if self.original_interaction.user.id in bot_state.active_interactions:
            print(f"[DEBUG @ 269] Timeout removing active interaction for user {self.original_interaction.user.id}")
            cleanup_interaction(self.original_interaction.user.id)
            await self.original_interaction.edit_original_response(
                embed=Embed(
                    title="⏱️ Selection Timed Out",
                    description="**You didn't respond in time!** Please run the command again.",
                    color=0xff0000
                ),
                view=None
            )
            return
         
        print(f"[DEBUG @ 274] Timeout ignored for user {self.original_interaction.user.id} since they interacted.")
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)
        return


    @discord.ui.button(emoji="1️⃣", label="Random Dataset", style=discord.ButtonStyle.primary)
    async def random_data_callback(self, interaction: discord.Interaction, button: Button):
        self.stop()
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        await interaction.response.send_message(
            f"{interaction.user.mention}, fetching a random dataset from Kaggle... ⏳", 
            ephemeral=True, delete_after=10
        )
        # dataset_prompt = await interaction.original_response()

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            def search_kaggle(query: str):
                """Searches Kaggle and returns the top 5 datasets."""
                datasets = api.dataset_list(search=query, file_type='csv', sort_by='votes')
                
                results = []
                for ds in datasets[:25]:
                    results.append({
                        "title": ds.title,
                        "ref": ds.ref,
                        # "size": ds.size
                    })
                return results

            print("Searching Kaggle for datasets...")
            datasets = search_kaggle("linear regression")
            if not datasets:
                await interaction.followup.send("No datasets found on Kaggle.", ephemeral=True)
                cleanup_interaction(interaction.user.id)
                return

            numeric_df = None
            dataset_ref = ""

            max_attempts = 5
            for _ in range(max_attempts):
                chosen_dataset = random.choice(datasets)
                dataset_ref = chosen_dataset['ref']

                with tempfile.TemporaryDirectory() as temp_dir:
                    api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
                    csv_files = glob.glob(os.path.join(temp_dir, '**', '*.csv'), recursive=True)
                    
                    if not csv_files:
                        continue 
                    
                    df = pd.read_csv(csv_files[0], engine="pyarrow")
                    
                    temp_numeric_df = get_numeric_dataframe(df)
                    
                    if temp_numeric_df.shape[1] >= 2:
                        numeric_df = temp_numeric_df
                        break

            if numeric_df is None:
                await interaction.followup.send("Could not find a random dataset with at least 2 numeric columns after 5 attempts. Try again!", ephemeral=True)
                cleanup_interaction(interaction.user.id)
                return

            if len(numeric_df) > 1000:
                numeric_df = numeric_df.sample(n=1000, random_state=42)

            if numeric_df.shape[1] > 25:
                max_columns_msg = await interaction.followup.send(
                    f"The dataset `{dataset_ref}` has >25 numeric columns. Using the first 25 for selection.",
                    ephemeral=True
                )
                numeric_df = numeric_df.iloc[:, :25]

            feature_view = DatasetView(numeric_df, 1)
            label_view = DatasetView(numeric_df, 2)
            
            feature_msg = await interaction.followup.send(
                content=f'**Dataset:** `{dataset_ref}`\nSelect the feature column to use as input.',
                view=feature_view,
                ephemeral=True
            )
            label_msg = await interaction.followup.send(
                content='Select the target column to predict.',
                view=label_view,
                ephemeral=True
            )

            try:
                selected_feature, selected_label = await asyncio.gather(
                    feature_view.selected_option,
                    label_view.selected_option
                )

            except asyncio.TimeoutError:
                await feature_msg.delete()
                await label_msg.delete()
                
                if 'max_columns_msg' in locals():
                    try:
                        await max_columns_msg.delete()
                    except Exception as e:
                        pass

                await self.original_interaction.edit_original_response(
                        embed=Embed(
                            title="⏱️ Timed Out! ⏱️",
                            description="**You didn't select a feature and label in time!** Please run the command again.",
                            color=0xff0000,
                        ),
                        view=None
                )
                cleanup_interaction(interaction.user.id)
                return None

            if selected_feature == selected_label:
                same_feature_label_msg = await interaction.followup.send('Feature and target columns must be different.', ephemeral=True)
                await same_feature_label_msg.delete(delay=5)
                cleanup_interaction(interaction.user.id)
                return None
            
            self.selected_data = df[[selected_feature, selected_label]]
            
            if 'max_columns_msg' in locals():
                try:
                    await max_columns_msg.delete()
                except Exception as e:
                    pass

            await feature_msg.delete()
            await label_msg.delete()
            # await dataset_prompt.delete()
            
            await calculate_linear_regression(interaction, self.selected_data, selected_feature, selected_label)
            cleanup_interaction(interaction.user.id)

        except Exception as e:
            await interaction.followup.send(f"An error occurred while fetching the dataset: {e}", ephemeral=True)
            cleanup_interaction(interaction.user.id)

    @discord.ui.button(emoji="2️⃣", label="Dataset File (CSV)", style=discord.ButtonStyle.primary)
    async def second_button_callback(self, interaction: discord.Interaction, button: Button):
        self.stop()  # Stop the view to prevent multiple interactions
        # if not interaction.permissions.attach_files:
        #     print("User lacks attach files permission, denying CSV upload.")
        #     await self.original_interaction.edit_original_response(
        #         embed=Embed(
        #             title="⚠️ Permission Error! ⚠️",
        #             description="**You lack the 'Attach Files' permission to do this.** Please run the command again.",
        #             color=0xff0000
        #         ),
        #         view=None
        #     )
        #     cleanup_interaction(interaction.user.id)
        #     # self.complete_selection(exc=PermissionError("User lacks Manage Messages permission."))
        #     return
        
        for item in self.children:
            item.disabled = True
        
        await self.original_interaction.edit_original_response(view=self)

        async def invalid_reply(message: discord.Message, reason: str):
            print(f"Invalid reply received: {message.content} - Reason: {reason}")
            if interaction.response.is_done():
                print("Interaction response already sent, using followup for invalid reply.")
                invalid_reply_msg = await interaction.followup.send(f"{message.author.mention}, {reason}", ephemeral=True)
                await invalid_reply_msg.delete(delay=5)
                try:
                    await message.delete()
                except discord.errors.Forbidden as e:
                    print("Failed to delete message due to lack of permissions or in DMs, skipping cleanup.")
                    # await interaction.followup.send(f"{message.author.mention}, I can't delete messages!", ephemeral=True)
                    pass
            else:
                print("Sending initial interaction response for invalid reply.")
                await interaction.response.send_message(f"{message.author.mention} {reason}", ephemeral=True, delete_after=5)
                # await invalid_reply_msg.delete(delay=5)
                try:
                    await message.delete()
                except discord.errors.Forbidden as e:
                    print("Failed to delete message due to lack of permissions or in DMs, skipping cleanup.")
                    # await interaction.followup.send(f"{message.author.mention}, I can't delete messages!", ephemeral=True)
                    pass

        if await interaction_perm_check(interaction):
            await self.original_interaction.edit_original_response(
                embed=Embed(title="How would you like to display your dataset?",
                            description="1️⃣ Random Dataset Generator \n 2️⃣ Dataset File \n 3️⃣ Given Values/Arrays"),
                view=self
            )
            await interaction.response.send_message(
                f"{interaction.user.mention}, please reply to this message with your dataset in a **CSV** file!")
            dataset_prompt = await interaction.original_response()

            async def run(m):
                await m.reply(f"Sorry, this command was not run by you! You can try it by running **/graph_linear_regression**!",
                              delete_after=5)
                await m.delete()
                return False

            async def media_reply(message: discord.Message, len_exceeded: bool, file_attached=True):
                if file_attached:
                    if len_exceeded:
                        asyncio.create_task(invalid_reply(message, 'There\'s too many files! Please upload just one CSV file!'))
                        return False
                    else:
                        asyncio.create_task(invalid_reply(message, 'Please upload just one CSV file.'))
                        # await message.delete()
                        return False
                else:
                    asyncio.create_task(invalid_reply(message, 'There\'s no files uploaded! Please upload just one CSV file!'))
                    # await message.delete()
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
                msg = await bot.wait_for("message", check=check, timeout=30.0)
                dataset_file = await msg.attachments[0].to_file()
                print("!")
                await dataset_prompt.delete()
                try:
                    await msg.delete()
                except discord.errors.Forbidden as e:
                    print("Failed to delete message due to lack of permissions or in DMs, skipping cleanup.")
                    # await interaction.followup.send(f"{interaction.user.mention}, I can't delete messages!", ephemeral=True)
                    pass
                df = pd.read_csv(dataset_file.fp, engine="pyarrow")

                numeric_df = get_numeric_dataframe(df)
                if numeric_df.shape[1] < 2:
                    await interaction.followup.send(
                        "Your CSV must include at least two numeric columns for feature and target selection.",
                        ephemeral=True
                    )
                    return
                if numeric_df.shape[1] > 25:
                    max_columns_msg = await interaction.followup.send(
                        "Your CSV has more than 25 numeric columns. Using the first 25 columns for selection.",
                        ephemeral=True
                    )
                    numeric_df = numeric_df.iloc[:, :25]

                feature_view = DatasetView(numeric_df, 1)
                label_view = DatasetView(numeric_df, 2)
                
                feature_msg = await interaction.followup.send(
                    content='Select the feature column to use as input.',
                    view=feature_view,
                    ephemeral=True
                )
                label_msg = await interaction.followup.send(
                    content='Select the target column to predict.',
                    view=label_view,
                    ephemeral=True
                )

                try:
                    selected_feature, selected_label = await asyncio.gather(
                        feature_view.selected_option,
                        label_view.selected_option
                    )

                except asyncio.TimeoutError:
                    print("User did not select feature/label in time.")
                    
                    await feature_msg.delete()
                    await label_msg.delete()

                    try:
                        await max_columns_msg.delete()
                    except Exception as e:
                        print(f"Failed to delete max_columns_msg: {e}")

                    await self.original_interaction.edit_original_response(
                            embed=Embed(
                                title="⏱️ Timed Out! ⏱️",
                                description="**You didn't select a feature and label in time!** Please run the command again.",
                                color=0xff0000,
                            ),
                            view=None
                    )
                    cleanup_interaction(interaction.user.id)
                    return None

                if selected_feature == selected_label:
                    await interaction.followup.send('Feature and target columns must be different.', ephemeral=True)
                    return None
                
                print("CHECK!")
                
                try:
                    await max_columns_msg.delete()
                except Exception as e:
                    print(f"Failed to delete max_columns_msg: {e}")

                await feature_msg.delete()
                await label_msg.delete()
                
                await linear_regression_calculator(interaction, numeric_df, selected_feature, selected_label)
                cleanup_interaction(interaction.user.id)
            except asyncio.TimeoutError:
                await self.original_interaction.edit_original_response(
                    embed=Embed(
                        title="⏱️ Selection Timed Out! ⏱️",
                        description="**You didn't respond in time!** Please run the command again.",
                        color=0xff0000
                    ),
                    view=None
                )
                print("Deleting dataset prompt...")
                await asyncio.create_task(safe_delete_message(dataset_prompt))
                # await interaction.delete_original_response()
                print(f"[DEBUG @ 380] Removed active interaction for user {interaction.user.id} due to invalid input.")
                cleanup_interaction(interaction.user.id)
            finally:
                pass

    
    @discord.ui.button(emoji="3️⃣", label="Manual Input", style=discord.ButtonStyle.primary)
    async def third_button_callback(self, interaction: discord.Interaction, button: Button):
        self.stop()
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        modal = ManualModal()
        await interaction.response.send_modal(modal)

        try:
            num_values = await asyncio.wait_for(modal.response_future, timeout=15)
            print(f"User entered: {num_values}")  # Optional logging
            # cleanup_interaction(interaction.user.id) 
        except asyncio.TimeoutError:
            await self.original_interaction.edit_original_response(
                embed=Embed(
                    title="⏱️ Modal Timed Out",
                    description="**You didn't respond in time!** Please run the command again.",
                    color=0xff0000,
                ),
                view=None
            )
            # self.complete_selection(exc=asyncio.TimeoutError("Selection timed out."))
            print(f"[DEBUG @ 289] Removed active interaction for user {interaction.user.id} due to invalid input.")
            cleanup_interaction(interaction.user.id)
        except ValueError:
            pass

async def calculate_neural_network(interaction: discord.Interaction, df: pd.DataFrame, feature_cols: list[str], label_col: str):
    """Calculates neural network regression, plots Actual vs Predicted in memory, and sends to Discord."""
    
    print(f"Calculating neural network for features {feature_cols} and label {label_col}...")
    await interaction.followup.send(
        f"Training Neural Network for **{feature_cols}** ➡️ **{label_col}**...", 
        ephemeral=True
    )

    def compute_and_plot():        
        plt.switch_backend('Agg')
        
        clean_df = df.dropna(subset=[feature_cols, label_col])
        
        X = clean_df[feature_cols].values.reshape(-1, 1)
        y = clean_df[label_col].values
        
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        nn = MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=150, random_state=42)
        nn.fit(X_scaled, y)
        
        predictions = nn.predict(X_scaled)
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.scatter(y, predictions, color="purple", alpha=0.5, label="Model Predictions")
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], color="k", linestyle="--", linewidth=2, label="Perfect Fit Line")
        
        ax1.set_xlabel(f"Actual {label_col}")
        ax1.set_ylabel(f"Predicted {label_col}")
        ax1.set_title(f"Prediction Accuracy: {label_col}")
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2.plot(nn.loss_curve_, color="blue", linewidth=2)
        ax2.set_xlabel("Training Iterations (Epochs)")
        ax2.set_ylabel("Loss (Error)")
        ax2.set_title("Network Learning Process (Loss Curve)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0) 
        plt.close()
        
        return buf, r2, mse

    try:
        loop = asyncio.get_running_loop()
        buf, r2, mse = await loop.run_in_executor(None, compute_and_plot)

        plot_file = File(fp=buf, filename="nn_regression.png")
        
        embed = Embed(
            title="🧠 Neural Network Results",
            description=f"**Target Variable:** `{label_col}`\n**R² Score:** `{r2:.4f}`\n**Mean Squared Error:** `{mse:.4f}`",
            color=0x9b59b6
        )
        embed.set_image(url="attachment://nn_regression.png")

        await interaction.followup.send(embed=embed, file=plot_file, ephemeral=True)
        
    except Exception as e:
        await interaction.followup.send(f"An error occurred during calculation: {e}", ephemeral=True)

class CreateNNView(View):
    def __init__(self, original_interaction):
        super().__init__(timeout=30)
        self.original_interaction = original_interaction
        self.selected_data = None 
    
    async def on_timeout(self):
        """Prevents timeout from removing an active interaction if the user has already interacted."""
        debug_values = []
        for v in bot_state.active_interactions.values():
            if isinstance(v, tuple):
                debug_values.append(v[0])
            else:
                debug_values.append(v)
        print(self.original_interaction, debug_values)

        for item in self.children:
            item.disabled = True
        
        if self.original_interaction.user.id in bot_state.active_interactions:
            print(f"[DEBUG @ 426] Timeout removing active interaction for user {self.original_interaction.user.id}")
            cleanup_interaction(self.original_interaction.user.id)
            await self.original_interaction.edit_original_response(
                embed=Embed(
                    title="⏱️ Selection Timed Out! ⏱️",
                    description="**You didn't respond in time!** Please run the command again.",
                    color=0xff0000
                ),
                view=None
            )
            return
        
        print(f"[DEBUG @ 431] Timeout ignored for user {self.original_interaction.user.id} since they interacted.")
        cleanup_interaction(self.original_interaction.user.id)
        await self.original_interaction.edit_original_response(
                embed=Embed(
                    title="⏱️ Selection Timed Out! ⏱️",
                    description="**You didn't respond in time!** Please run the command again.",
                    color=0xff0000
                ),
                view=None
        )
        # await self.original_interaction.edit_original_response(view=None)
        return


    @discord.ui.button(emoji="1️⃣", label="Random Dataset", style=discord.ButtonStyle.primary)
    async def random_data_callback(self, interaction: discord.Interaction, button: Button):
        self.stop()
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        await interaction.response.send_message(
            f"{interaction.user.mention}, fetching a random dataset from Kaggle... ⏳\n*(This takes a moment to download and unzip in the background!)*", 
            ephemeral=True, delete_after=10
        )

        loop = asyncio.get_running_loop()

        def fetch_and_process_dataset():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            search_topics = [
                "regression", 
                "salary predict", 
                "housing prices", 
                "insurance costs", 
                "sales forecasting",
                "weather numerical"
            ]
            query = random.choice(search_topics)

            datasets = api.dataset_list(max_size=5_000_000, file_type='csv', sort_by='votes', search=query)
            results = list(datasets[:25])

            if not results:
                return None, None, None, "No datasets found on Kaggle."

            max_attempts = 5
            for _ in range(max_attempts):
                chosen_dataset = random.choice(results)
                dataset_ref = chosen_dataset.ref

                with tempfile.TemporaryDirectory() as temp_dir:
                    api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
                    csv_files = glob.glob(os.path.join(temp_dir, '**', '*.csv'), recursive=True)
                    
                    if not csv_files:
                        continue 
                    
                    df = pd.read_csv(csv_files[0], engine="pyarrow")
                    temp_numeric_df = get_numeric_dataframe(df)
                    
                    if temp_numeric_df.shape[1] >= 2:
                        return df, temp_numeric_df, dataset_ref, None

            return None, None, None, "Could not find a random dataset with at least 2 numeric columns after 5 attempts. Try again!"

        try:
            df, numeric_df, dataset_ref, error_msg = await loop.run_in_executor(None, fetch_and_process_dataset)

            if error_msg:
                await interaction.followup.send(error_msg, ephemeral=True)
                cleanup_interaction(interaction.user.id)
                return

            if len(numeric_df) > 1000:
                numeric_df = numeric_df.sample(n=1000, random_state=42)

            if numeric_df.shape[1] > 25:
                max_columns_msg = await interaction.followup.send(
                    f"The dataset `{dataset_ref}` has >25 numeric columns. Using the first 25 for selection.",
                    ephemeral=True
                )
                numeric_df = numeric_df.iloc[:, :25]

            feature_view = DatasetView(numeric_df, 1)
            label_view = DatasetView(numeric_df, 2) 
            
            feature_msg = await interaction.followup.send(
                content=f'**Dataset:** `{dataset_ref}`\nSelect the feature column to use as input.',
                view=feature_view,
                ephemeral=True
            )
            label_msg = await interaction.followup.send(
                content='Select the target column to predict.',
                view=label_view,
                ephemeral=True
            )

            try:
                selected_feature, selected_label = await asyncio.gather(
                    feature_view.selected_option,
                    label_view.selected_option
                )

            except asyncio.TimeoutError:
                await feature_msg.delete()
                await label_msg.delete()
                
                if 'max_columns_msg' in locals():
                    try:
                        await max_columns_msg.delete()
                    except Exception as e:
                        pass

                await self.original_interaction.edit_original_response(
                        embed=Embed(
                            title="⏱️ Timed Out! ⏱️",
                            description="**You didn't select a feature and label in time!** Please run the command again.",
                            color=0xff0000,
                        ),
                        view=None
                )
                cleanup_interaction(interaction.user.id)
                return None

            if selected_feature == selected_label:
                same_feature_label_msg = await interaction.followup.send('Feature and target columns must be different.', ephemeral=True)
                await same_feature_label_msg.delete(delay=5)
                cleanup_interaction(interaction.user.id)
                return None
            
            self.selected_data = df[[selected_feature, selected_label]]
            
            if 'max_columns_msg' in locals():
                try:
                    await max_columns_msg.delete()
                except Exception as e:
                    pass

            await feature_msg.delete()
            await label_msg.delete()
            
            await calculate_neural_network(interaction, self.selected_data, selected_feature, selected_label)
            cleanup_interaction(interaction.user.id)

        except Exception as e:
            await interaction.followup.send(f"An error occurred while fetching the dataset: {e}", ephemeral=True)
            cleanup_interaction(interaction.user.id)

    @discord.ui.button(emoji="2️⃣", label="Dataset File", style=discord.ButtonStyle.primary)
    async def upload_csv_callback(self, interaction: discord.Interaction, button: Button):
        self.stop()  # Stop the view to prevent multiple interactions
        # if not interaction.permissions.attach_files:
        #     print("User lacks attach files permission, denying CSV upload.")
        #     await self.original_interaction.edit_original_response(
        #         embed=Embed(
        #             title="⚠️ Permission Error! ⚠️",
        #             description="**You lack the 'Attach Files' permission to do this.** Please run the command again.",
        #             color=0xff0000
        #         ),
        #         view=None
        #     )
        #     cleanup_interaction(interaction.user.id)
        #     # self.complete_selection(exc=PermissionError("User lacks Manage Messages permission."))
        #     return
        
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        async def invalid_reply(message: discord.Message, reason: str):
            if interaction.response.is_done():
                invalid_reply_msg = await interaction.followup.send(f"{message.author.mention}, {reason}", ephemeral=True)
                await invalid_reply_msg.delete(delay=5)
                try:
                    await message.delete()
                except discord.errors.Forbidden as e:
                    print("Failed to delete message due to lack of permissions or in DMs, skipping cleanup.")
                    # await interaction.followup.send(f"{message.author.mention}, I can't delete messages!", ephemeral=True)
                    pass
            else:
                print("Sending initial interaction response for invalid reply.")
                await interaction.response.send_message(f"{message.author.mention} {reason}", ephemeral=True, delete_after=5)
                # await invalid_reply_msg.delete(delay=5)
                try:
                    await message.delete()
                except discord.errors.Forbidden as e:
                    print("Failed to delete message due to lack of permissions or in DMs, skipping cleanup.")
                    # await interaction.followup.send(f"{message.author.mention}, I can't delete messages!", ephemeral=True)
                    pass


        if await interaction_perm_check(interaction):
            await interaction.response.send_message(
                f"{interaction.user.mention}, please reply to this message with your dataset in a **CSV** file!")
            dataset_prompt = await interaction.original_response()

            async def run(m):
                await m.reply(f"Sorry, this command was not run by you! You can try it by running **/create_neural_network**!",
                              delete_after=5)
                await m.delete()
                return False

            async def media_reply(message: discord.Message, len_exceeded: bool, file_attached=True):
                if file_attached:
                    if len_exceeded:
                        asyncio.create_task(invalid_reply(message, 'There\'s too many files! Please upload just one CSV file!'))
                        return False
                    else:
                        asyncio.create_task(invalid_reply(message, 'Please upload just one CSV file.'))
                        # await message.delete()
                        return False
                else:
                    asyncio.create_task(invalid_reply(message, 'There\'s no files uploaded! Please upload just one CSV file!'))
                    # await message.delete()
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
                msg = await bot.wait_for("message", check=check, timeout=30.0)
                dataset_file = await msg.attachments[0].to_file()
                await dataset_prompt.delete()
                try:
                    await msg.delete()
                except discord.errors.Forbidden as e:
                    if interaction.guild is None:
                        print("Failed to delete message in DMs, skipping cleanup.")
                    else:
                        await interaction.followup.send(f"{interaction.user.mention}, I can't delete messages!", ephemeral=True)
                    # print("Failed to delete message due to lack of permissions or in DMs, skipping cleanup.")
                    pass
                df = pd.read_csv(dataset_file.fp, engine="pyarrow")
                numeric_df = get_numeric_dataframe(df)
                if numeric_df.shape[1] < 2:
                    await interaction.followup.send(
                        'Your dataset must contain at least two numeric columns for feature and label selection.',
                        ephemeral=True
                    )
                    return
                if numeric_df.shape[1] > 25:
                    max_columns_msg = await interaction.followup.send(
                        'Your dataset has more than 25 numeric columns. Using the first 25 columns for selection.',
                        ephemeral=True
                    )
                    numeric_df = numeric_df.iloc[:, :25]

                feature_view = DatasetView(numeric_df, 1)
                label_view = DatasetView(numeric_df, 2)
                feature_msg = await interaction.followup.send(
                    content='Select the feature column to use as input.',
                    view=feature_view,
                    ephemeral=True
                )
                label_msg = await interaction.followup.send(
                    content='Select the target column to predict.',
                    view=label_view,
                    ephemeral=True
                )

                try:
                    selected_feature, selected_label = await asyncio.gather(
                        feature_view.selected_option,
                        label_view.selected_option
                    )

                except asyncio.TimeoutError:
                    print("User did not select feature/label in time.")
                    
                    await feature_msg.delete()
                    await label_msg.delete()

                    try:
                        await max_columns_msg.delete()
                    except Exception as e:
                        print(f"Failed to delete max_columns_msg: {e}")

                    await self.original_interaction.edit_original_response(
                            embed=Embed(
                                title="⏱️ Selection Timed Out! ⏱️",
                                description="**You didn't select a feature and label in time!** Please run the command again.",
                                color=0xff0000,
                            ),
                            view=None
                    )
                    cleanup_interaction(interaction.user.id)
                    return None

                if selected_feature == selected_label:
                    await interaction.followup.send('Feature and target columns must be different.', ephemeral=True)
                    return None

                self.selected_data = df[[selected_feature, selected_label]]
                print("CHECK!")
                try:
                    await max_columns_msg.delete()
                except Exception as e:
                    print(f"Failed to delete max_columns_msg: {e}")
                await feature_msg.delete()
                await label_msg.delete()
                await calculate_neural_network(interaction, self.selected_data, selected_feature, selected_label)
                cleanup_interaction(interaction.user.id)
            except asyncio.TimeoutError:
                await asyncio.create_task(safe_delete_message(dataset_prompt))
                await self.original_interaction.edit_original_response(
                    embed=Embed(
                        title="⏱️ Selection Timed Out! ⏱️",
                        description="**You didn't respond in time!** Please run the command again.",
                        color=0xff0000
                    ),
                    view=None
                )
                # await interaction.delete_original_response()
                print(f"[DEBUG @ 508] Removed active interaction for user {interaction.user.id} due to invalid input.")
                cleanup_interaction(interaction.user.id)

    async def start_training(self, interaction: discord.Interaction, dataframe: pd.DataFrame, feature_cols, label_col):
        """Handles the training after dataset selection."""
        self.stop()
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        await interaction.followup.send(
            "Training the neural network with your selected dataset. This may take a while...",
            ephemeral=True,
        )

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            model, history, scaler = await loop.run_in_executor(pool, train_neural_network, dataframe, feature_cols, label_col, None)

        if model and history:
            try:
                files = []
                
                if os.path.exists('model_architecture.png'):
                    files.append(File('model_architecture.png', filename='neural_network_architecture.png'))
                
                embed = Embed(
                    title="🧠 Neural Network Training Complete!",
                    description=f"Trained on {len(dataframe)} samples.\nFeatures: {', '.join(feature_cols)}\nLabel: {label_col}",
                    color=0x00ff00
                )
                
                if 'accuracy' in history.history:
                    embed.add_field(name="Training Accuracy", value=f"{history.history['accuracy'][-1]:.4f}", inline=True)
                    embed.add_field(name="Validation Accuracy", value=f"{history.history['val_accuracy'][-1]:.4f}", inline=True)
                else:
                    embed.add_field(name="Training MAE", value=f"{history.history['mae'][-1]:.4f}", inline=True)
                    embed.add_field(name="Validation MAE", value=f"{history.history['val_mae'][-1]:.4f}", inline=True)
                
                await interaction.followup.send(embed=embed, files=files, ephemeral=True)
                
                # Cache the model
                bot_state.nn_model_cache = bot_state.nn_model_cache or {}
                bot_state.nn_model_cache[interaction.user.id] = {
                    'model': model,
                    'feature_cols': feature_cols,
                    'label_col': label_col,
                    'scaler': scaler
                }

                cleanup_interaction(interaction.user.id)
                
            except Exception as e:
                await interaction.followup.send(f"Error sending results: {e}", ephemeral=True)
        else:
            await interaction.followup.send("Training failed. Please check the logs.", ephemeral=True)
            
        print(f"[DEBUG @ 548] Removed active interaction for user {interaction.user.id} due to invalid input.")
        cleanup_interaction(interaction.user.id)

    @discord.ui.button(emoji="3️⃣", label="Manual Input", style=discord.ButtonStyle.primary)
    async def manual_input_callback(self, interaction: discord.Interaction, button: Button):
        """Let user manually input data values."""
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        modal = ManualModal()
        await interaction.response.send_modal(modal)
        # bot_state.active_interactions[interaction.user.id] = interaction.response

async def interaction_perm_check(interaction: discord.Interaction):
    if interaction.guild is None:
        # await interaction.response.send_message("This command can only be used in a server channel.", ephemeral=True)
        return True
    interaction_user_perms = interaction.channel.permissions_for(interaction.user)
    interaction_bot_perms = interaction.channel.permissions_for(interaction.guild.me)
    print(f"User perms: {interaction_user_perms}, Bot perms: {interaction_bot_perms}")
    if interaction_user_perms.attach_files and interaction_bot_perms.attach_files:
        return True
    elif not interaction_user_perms.attach_files:
        await interaction.response.send_message(
            "You lack the **'Attach Files'** permission to upload a CSV file! Please run the command again in a channel where you have that permission, or direct message me!",
            ephemeral=True, delete_after=10)
        print(f"[DEBUG @ 535] Removed active interaction for user {interaction.user.id} due to invalid input.")
        cleanup_interaction(interaction.user.id)
        # bot_state.graphlr_active_interactions.discard(interaction)
        return False
    elif not interaction_bot_perms.attach_files:
        await interaction.response.send_message(
            "I lack the **'Attach Files'** permission to receive a CSV file! Please run the command again in a channel where I have that permission, or direct message me!",
            ephemeral=True, delete_after=10)
        print(f"[DEBUG @ 535] Removed active interaction for user {interaction.user.id} due to invalid input.")
        cleanup_interaction(interaction.user.id)
        # bot_state.graphlr_active_interactions.discard(interaction)
        return False
    else:
        await interaction.response.send_message(
            "We can't upload files in this channel if we want to complete the interaction! Please move to a **different channel** or **direct message** me!",
            ephemeral=True)
        print(f"[DEBUG @ 537] Removed active interaction for user {interaction.user.id} due to invalid input.")
        cleanup_interaction(interaction.user.id)
        # bot_state.graphlr_active_interactions.discard(interaction)
        return False

async def remove_after_timeout(interaction: discord.Interaction, user_id, delay):
    try:
        await asyncio.sleep(delay)
        
        current_value = bot_state.active_interactions.get(user_id)
        if current_value and isinstance(current_value, tuple):
            tracked_interaction, _ = current_value
            
            if tracked_interaction.id == interaction.id:
                await interaction.followup.send(
                    "⏱️ Your interaction has timed out due to inactivity. Please run the command again if you'd like to try again!",
                    ephemeral=True
                )
                bot_state.active_interactions.pop(user_id, None)
                
    except asyncio.CancelledError:
        pass

def cleanup_interaction(user_id):
    """Cleanup user interaction after visualization is sent."""
    value = bot_state.active_interactions.pop(user_id, None)
    
    if value:
        if isinstance(value, tuple):
            _, task = value
            task.cancel()
        else:
            asyncio.create_task(safe_delete_message(value))


async def safe_delete_message(message):
    """Safely delete a message without raising exceptions."""
    try:
        await message.delete()
    except Exception:
        pass

async def check_user_instances(interaction: discord.Interaction):
    """Checks if a user has an ongoing interaction and prevents new ones."""
    user_id = interaction.user.id
    lock = bot_state.get_user_lock(user_id)
    print(f"[DEBUG] Acquiring lock for user {user_id} to check active interactions.")
    async with lock:
        print(f"[DEBUG] Checking active interactions for user {user_id}. Current active interactions: {list(bot_state.active_interactions.keys())}")
        if user_id in bot_state.active_interactions.keys():
            await interaction.response.send_message(
                "You already have an active interaction! Please complete it before starting a new one.",
                ephemeral=True, delete_after=10
            )
            return False    
        
        task = asyncio.create_task(remove_after_timeout(interaction, user_id, 300))
        bot_state.active_interactions[user_id] = (interaction, task)
        return True


@bot.tree.command(name="describe_data", description="Generate a quick dataset summary and correlation heatmap.")
@app_commands.check(initialization_check)
# @app_commands.checks.bot_has_permissions(view_channel=True, send_messages=True, attach_files=True, add_reactions=True, manage_messages=True)
# @app_commands.checks.has_permissions(view_channel=True, send_messages=True, use_application_commands=True)
async def describe_data(interaction: discord.Interaction):
    # if interaction.guild is None:
        # await interaction.response.send_message("This command can only be used in a server channel.", ephemeral=True)
        # return True
    # interaction_user_perms = interaction.channel.permissions_for(interaction.user)
    if not await check_user_instances(interaction):
        # print(interaction_bot_perms.attach_files)
        return
    else:
        # print(f"[DEBUG @ 2602] Starting graph_linear_regression command for user {interaction.user.id}")
        pass
    #     bot_state.active_interactions[interaction.user.id] = interaction
    #     print(bot_state.active_interactions)
    
    await interaction.response.defer(ephemeral=True)

    if interaction.guild:
        interaction_bot_perms = interaction.channel.permissions_for(interaction.guild.me)
        if not interaction_bot_perms.attach_files:
            print("Bot lacks attach files permission, denying CSV upload.")
            await interaction.edit_original_response(
                    embed=Embed(
                        title="⚠️ Permission Error! ⚠️",
                        description=f"Please run the command again in a channel where I have that permission, or direct message me!",
                        color=0xff0000
                    ),
                    view=None
                )
            # await interaction.edit_original_response(
            #     "I lack the **'Attach Files'** permission to receive a CSV file! Please run the command again in a channel where I have that permission, or direct message me!")
            print(f"[DEBUG @ 535] Removed active interaction for user {interaction.user.id} due to invalid input.")
            cleanup_interaction(interaction.user.id)
                # bot_state.graphlr_active_interactions.discard(interaction)
            return
    else:
        print("Interaction is in DMs, skipping permission check.")
    
    try:
        await interaction.response.defer(ephemeral=True)
    except Exception as e:
        print(f"[ERROR] Failed to defer interaction response for user {interaction.user.id}: {e}")

    try:
        df = await ask_for_dataset_via_menu(
            interaction,
            title="How would you like to provide your dataset?",
            description="1️⃣ Random Dataset Generator \n2️⃣ Dataset File \n3️⃣ Manual Input"
        )
        if df is None:
            # cleanup_interaction(interaction.user.id)
            return

        embed = await asyncio.get_event_loop().run_in_executor(None, build_dataset_summary_embed, df)
        print(f"[DEBUG @ 610] Built dataset summary embed for user {interaction.user.id}")
        files = []
        heatmap_saved = await asyncio.get_event_loop().run_in_executor(None, save_correlation_heatmap, df)
        if heatmap_saved:
            files.append(File('correlation_heatmap.png'))
        await interaction.followup.send(embed=embed, files=files, ephemeral=True)
    finally:
        cleanup_interaction(interaction.user.id)



@bot.tree.command(name="compare_models", description="Compare multiple regression models on a cached dataset.")
@app_commands.check(initialization_check)
# @app_commands.checks.bot_has_permissions(view_channel=True, send_messages=True, attach_files=True, add_reactions=True, manage_messages=True)
# @app_commands.checks.has_permissions(view_channel=True, send_messages=True, use_application_commands=True)
async def compare_models(interaction: discord.Interaction):
        # await interaction.response.send_message("This command can only be used in a server channel.", ephemeral=True)
        # return True
    # interaction_user_perms = interaction.channel.permissions_for(interaction.user)
    if not await check_user_instances(interaction):
        # print(interaction_bot_perms.attach_files)
        return
    else:
        print(f"[DEBUG @ 2603] Starting compare_models command for user {interaction.user.id}")
    #     bot_state.active_interactions[interaction.user.id] = interaction
    
    await interaction.response.defer(ephemeral=True)
    
    if interaction.guild:
        interaction_bot_perms = interaction.channel.permissions_for(interaction.guild.me)
        if not interaction_bot_perms.attach_files:
            print("Bot lacks attach files permission, denying CSV upload.")
            await interaction.edit_original_response(
                    embed=Embed(
                        title="⚠️ Permission Error! ⚠️",
                        description=f"Please run the command again in a channel where I have that permission, or direct message me!",
                        color=0xff0000
                    ),
                    view=None
                )
            # await interaction.edit_original_response(
            #     "I lack the **'Attach Files'** permission to receive a CSV file! Please run the command again in a channel where I have that permission, or direct message me!")
            print(f"[DEBUG @ 535] Removed active interaction for user {interaction.user.id} due to invalid input.")
            cleanup_interaction(interaction.user.id)
                # bot_state.graphlr_active_interactions.discard(interaction)
            return

    try:
        await interaction.response.defer(ephemeral=True)
    except Exception as e:
        print(f"[ERROR] Failed to defer interaction response for user {interaction.user.id}: {e}")

    try:
        df = await ask_for_dataset_via_menu(
            interaction,
            title="How would you like to provide your dataset?",
            description="1️⃣ Random Dataset Generator \n2️⃣ Dataset File \n3️⃣ Manual Input"
        )
        if df is None:
            return
            cache_dataset(interaction.user.id, df)

        if list(df.columns) == ['feature', 'label']:
            feature_col, label_col = 'feature', 'label'
        else:
            try:
                print(f"[DEBUG @ 617] Starting feature/label selection for user {interaction.user.id}")
                selected = await select_feature_and_label(interaction, df)
                if selected is None:
                    return
                feature_col, label_col = selected
            except Exception as exc:
                await interaction.followup.send(f'Selection failed: {exc}', ephemeral=True)
                print(f"[DEBUG @ 617] Removed active interaction for user {interaction.user.id} due to invalid input.")
                cleanup_interaction(interaction.user.id)
                return

        try:
            results = await asyncio.get_event_loop().run_in_executor(None, compare_models_on_dataframe, df, feature_col, label_col)
            await interaction.followup.send('Training models and comparing performance. This may take a moment...', ephemeral=True)
            embed = compare_models_embed(results, feature_col, label_col)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except ValueError as exc:
            print(f"Error during model comparison for user {interaction.user.id}: {exc}")
            await interaction.edit_original_response(
                embed=Embed(
                    title="⚠️ Invalid Input! ⚠️",
                    description=f"Please ensure your dataset has more than **5** numeric feature and label columns!",
                    color=0xff0000
                ),
                view=None
            )
        except TimeoutError as exc:
            # await interaction.edit_original_response(f'Comparison failed: {exc}', ephemeral=True)
            await interaction.edit_original_response(
                embed=Embed(
                    title="⏱️ Timed Out! ⏱️",
                    description="**You didn't select a feature and label in time!** Please run the command again.",
                    color=0xff0000,
                ),
                view=None
            )
    finally:
        cleanup_interaction(interaction.user.id)



@bot.command(name="restart", description="Restarts the bot if issues are encountered.")
@commands.is_owner()
async def restart(ctx: discord.ext.commands.Context):
    await ctx.reply("Restarting bot...")
    os.execv(sys.executable, ["python"] + sys.argv)


@bot.tree.command(name="graph_linear_regression", description="Graphs a linear regression model of the given dataset/values.")
@app_commands.check(initialization_check)
# @app_commands.checks.bot_has_permissions(view_channel=True, send_messages=True, attach_files=True, add_reactions=True, manage_messages=True)
# @app_commands.checks.has_permissions(view_channel=True, send_messages=True, use_application_commands=True)
async def graph_linear_regression(interaction: discord.Interaction):
    # if interaction.guild is None:
        # await interaction.response.send_message("This command can only be used in a server channel.", ephemeral=True)
        # return True
    # interaction_user_perms = interaction.channel.permissions_for(interaction.user)
    # interaction_bot_perms = interaction.channel.permissions_for(interaction.guild.me)
    if not await check_user_instances(interaction):
        # print(interaction_bot_perms.attach_files)
        return
    else:        
        print(f"[DEBUG @ 2602] Starting graph_linear_regression command for user {interaction.user.id}")
    #     bot_state.active_interactions[interaction.user.id] = interaction
    
    await interaction.response.defer(ephemeral=True)

    if interaction.guild:
        interaction_bot_perms = interaction.channel.permissions_for(interaction.guild.me)
        if not interaction_bot_perms.attach_files:
            print("Bot lacks attach files permission, denying CSV upload.")
            await interaction.edit_original_response(
                    embed=Embed(
                        title="⚠️ Permission Error! ⚠️",
                        description=f"Please run the command again in a channel where I have 'Attach Files' permission, or direct message me!",
                        color=0xff0000
                    ),
                    view=None
                )
            # await interaction.edit_original_response(
            #     "I lack the **'Attach Files'** permission to receive a CSV file! Please run the command again in a channel where I have that permission, or direct message me!")
            print(f"[DEBUG @ 535] Removed active interaction for user {interaction.user.id} due to invalid input.")
            cleanup_interaction(interaction.user.id)
                # bot_state.graphlr_active_interactions.discard(interaction)
            return
    # await interaction.response.defer(ephemeral=True)
    try:
        graphlr_view = GraphLRView(interaction)
        await interaction.followup.send(embed=Embed(title="How would you like to display your dataset?",
                                                    description="1️⃣ Random Dataset Generator \n 2️⃣ Dataset File \n 3️⃣ Given Numbers/Array"),
                                        view=graphlr_view, ephemeral=True)
    except TimeoutError as e:
        print(f"[DEBUG @ 535] Timeout error occurred for user {interaction.user.id}: {e}")
        await interaction.edit_original_response(
            content="Command timed out. Please try again."
        )
        cleanup_interaction(interaction.user.id)


@bot.tree.command(name="create_neural_network", description="Creates a neural network model of the given dataset/values.")
@app_commands.check(initialization_check)
# @app_commands.checks.bot_has_permissions(view_channel=True, send_messages=True, attach_files=True, add_reactions=True, manage_messages=True)
# @app_commands.checks.has_permissions(view_channel=True, send_messages=True, use_application_commands=True)
async def create_neural_network(interaction: discord.Interaction):
    # if interaction.guild is None:
        # await interaction.response.send_message("This command can only be used in a server channel.", ephemeral=True)
        # return True
    # interaction_user_perms = interaction.channel.permissions_for(interaction.user)
    # interaction_bot_perms = interaction.channel.permissions_for(interaction.guild.me)
    # print(f"User perms: {interaction.channel.permissions_for(interaction.user.id)}, Bot perms: {interaction_bot_perms}")
    if not await check_user_instances(interaction):
        # print(interaction_bot_perms.attach_files)
        return
    else:
        print(f"[DEBUG @ 2603] Starting create_neural_network command for user {interaction.user.id}")
    #     bot_state.active_interactions[interaction.user.id] = interaction

    await interaction.response.defer(ephemeral=True)

    if interaction.guild:
        interaction_bot_perms = interaction.channel.permissions_for(interaction.guild.me)
        if not interaction_bot_perms.attach_files:
            print("Bot lacks attach files permission, denying CSV upload.")
            await interaction.edit_original_response(
                    embed=Embed(
                        title="⚠️ Permission Error! ⚠️",
                        description=f"Please run the command again in a channel where I have that permission, or direct message me!",
                        color=0xff0000
                    ),
                    view=None
                )
            # await interaction.edit_original_response(
            #     "I lack the **'Attach Files'** permission to receive a CSV file! Please run the command again in a channel where I have that permission, or direct message me!")
            print(f"[DEBUG @ 535] Removed active interaction for user {interaction.user.id} due to invalid input.")
            cleanup_interaction(interaction.user.id)
                # bot_state.graphlr_active_interactions.discard(interaction)
            return

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
        print(f"[DEBUG @ 2604] Sent dataset selection menu for neural network creation to user {interaction.user.id}")
        # cleanup_interaction(interaction.user.id)
    except Exception as e:
        print(f"Error in create_neural_network: {e}")
        await interaction.followup.send("An error occurred while processing your request.", ephemeral=True)

# @bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    print(f"[DEBUG] Handling error for user {interaction.user.id}: {error}")
    if isinstance(error, app_commands.CommandInvokeError):
        original = error.original
        if isinstance(original, discord.errors.NotFound) and original.code == 10062:
            print(f"[WARNING] Interaction timed out for {interaction.user.name}. Aborting error response.")
            # cleanup_interaction(interaction.user.id)
            return
        else:
            msg = f"An internal error occurred: {original}"
            print(f"[ERROR] In command {interaction.command.name}: {original}")

    elif isinstance(error, app_commands.MissingPermissions) and isinstance(error, app_commands.BotMissingPermissions):
        missing_user = ", ".join(error.missing_permissions)
        missing_bot = ", ".join(error.missing_permissions)
        msg = f"❌ You are missing the following permissions: `{missing_user}`\n❌ I am missing the following permissions to execute this: `{missing_bot}`"

    elif isinstance(error, app_commands.MissingPermissions):
        missing = ", ".join(error.missing_permissions)
        msg = f"❌ You are missing the following permissions: `{missing}`"
        
    elif isinstance(error, app_commands.BotMissingPermissions):
        missing = ", ".join(error.missing_permissions)
        msg = f"❌ I am missing the following permissions to execute this: `{missing}`"


    elif isinstance(error, app_commands.CheckFailure):
        msg = "Initialization is not complete. Please try again later."

    else:
        msg = "An unexpected error occurred while processing the command."
        print(f"[ERROR] Unhandled exception: {error}")

    try:
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True, delete_after=10)
    except discord.errors.NotFound:
        pass

    cleanup_interaction(interaction.user.id)

# @bot.event
# async def on_command_error(ctx, exception):
#     if isinstance(exception, commands.CommandOnCooldown):
#         await ctx.reply(f"You are rate limited. Please, try again in {exception.retry_after} seconds")
#     elif isinstance(exception, commands.CheckFailure):
#         await ctx.reply("You don't have the necessary permissions to run this command!")
#     else:
#         print(f"Unhandled exception: {exception}")


# @graph_linear_regression.error
# async def graph_linear_regression_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
#     if isinstance(error, app_commands.CheckFailure):
#         await interaction.response.send_message("Initialization is not complete. Please try again later.",
#                                                 ephemeral=True)
#     else:
#         print(f"[DEBUG @ 617] Removed active interaction for user {interaction.user.id} due to invalid input.")
#         cleanup_interaction(interaction.user.id)
#         await interaction.response.send_message("An error occurred while processing the command.", ephemeral=True)

# @create_neural_network.error
# async def create_neural_network_error(interaction: discord.Interaction, error: app_commands.AppCommandError):

#     print(f"[DEBUG @ 617] Handling error for user {interaction.user.id}: {error}")
    
#     # 1. Determine the correct error message FIRST (Specific to Broad)
#     if isinstance(error, app_commands.MissingPermissions):
#         missing = ", ".join(error.missing_permissions)
#         msg = f"❌ You are missing the following permissions to run this command: `{missing}`"
        
#     elif isinstance(error, app_commands.BotMissingPermissions):
#         print(f"[DEBUG] BotMissingPermissions: {error.missing_permissions}")
#         missing = ", ".join(error.missing_permissions)
#         msg = f"❌ I am missing the following permissions to execute this command: `{missing}`"
        
#     # CheckFailure goes LAST because it acts as the catch-all for custom checks
#     elif isinstance(error, app_commands.CheckFailure):
#         msg = "Initialization is not complete. Please try again later."
        
#     else:
#         print(f"[DEBUG] Error: {error} | Removed active interaction for user {interaction.user.id}")
#         cleanup_interaction(interaction.user.id)
#         msg = "An error occurred while processing the command."

#     # 2. Safely attempt to send the message
#     try:
#         print(f"[DEBUG] Attempting to send error message to {interaction.user.name}: {msg}")
#         if interaction.response.is_done():
#             # If already deferred/responded, use followup
#             print(f"[DEBUG] Using followup to send message to {interaction.user.name}")
#             await interaction.followup.send(msg, ephemeral=True)
#         else:
#             print(f"[DEBUG] Using standard response to send message to {interaction.user.name}")
#             # If not responded to yet, use standard response
#             await interaction.response.send_message(msg, ephemeral=True)
            
#     except discord.errors.NotFound:
#         # If we still get a 10062 error here, the interaction completely timed out (>3 seconds).
#         # We catch it silently so it doesn't clutter the console traceback.
#         print(f"[WARNING] Could not send error message to {interaction.user.name}. In/teraction timed out.")

# @compare_models.error
# async def compare_models_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
#     if isinstance(error, app_commands.CheckFailure):
#         await interaction.response.send_message("Initialization is not complete. Please try again later.",
#                                                 ephemeral=True)
#     else:
#         print(f"[DEBUG @ 628] Removed active interaction for user {interaction.user.id} due to invalid input.")
#         cleanup_interaction(interaction.user.id)
#         await interaction.response.send_message("An error occurred while processing the command.", ephemeral=True)

# @describe_data.error
# async def describe_data_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
#     if isinstance(error, app_commands.CheckFailure):
#         await interaction.response.send_message("Initialization is not complete. Please try again later.",
#                                                 ephemeral=True)
#     else:
#         print(f"[DEBUG @ 628] Removed active interaction for user {interaction.user.id} due to invalid input.")
#         cleanup_interaction(interaction.user.id)
#         await interaction.response.send_message("An error occurred while processing the command.", ephemeral=True)

print(os.getenv('KAGGLE_USERNAME'), os.getenv('KAGGLE_KEY'))
keep_alive()
bot.run(token)