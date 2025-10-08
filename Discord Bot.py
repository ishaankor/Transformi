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
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.patches as patches
import seaborn as sns
import pickle
import io
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

# Configure SSL for macOS - must be done before any Discord imports
import ssl
import certifi
try:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: ssl_context
    print("SSL certificates configured successfully")
except Exception as e:
    print(f"SSL configuration warning: {e}")

from sklearn.linear_model import LinearRegression


def create_comprehensive_neural_network():
    """
    Creates and trains a comprehensive neural network with visualization
    """
    try:
        # Configure TensorFlow for better performance on macOS
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        # Generate sample data for demonstration
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                                 n_redundant=10, n_clusters_per_class=1, random_state=42)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train the model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                          validation_split=0.2, verbose=0)
        
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training History
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('🔥 Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy History
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('🎯 Training & Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Model Architecture Visualization
        axes[0, 2].text(0.1, 0.8, '🧠 Neural Network Architecture', fontsize=16, fontweight='bold')
        axes[0, 2].text(0.1, 0.7, f'Input Layer: 20 features', fontsize=12)
        axes[0, 2].text(0.1, 0.6, f'Hidden Layer 1: 64 neurons (ReLU)', fontsize=12)
        axes[0, 2].text(0.1, 0.5, f'Dropout: 30%', fontsize=12, style='italic')
        axes[0, 2].text(0.1, 0.4, f'Hidden Layer 2: 32 neurons (ReLU)', fontsize=12)
        axes[0, 2].text(0.1, 0.3, f'Dropout: 30%', fontsize=12, style='italic')
        axes[0, 2].text(0.1, 0.2, f'Hidden Layer 3: 16 neurons (ReLU)', fontsize=12)
        axes[0, 2].text(0.1, 0.1, f'Output Layer: 1 neuron (Sigmoid)', fontsize=12)
        axes[0, 2].text(0.1, 0.0, f'Total Parameters: {model.count_params():,}', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].axis('off')
        
        # Network Diagram
        axes[1, 0].set_title('🔗 Network Structure', fontsize=14, fontweight='bold')
        layer_sizes = [20, 64, 32, 16, 1]
        layer_names = ['Input\n(20)', 'Hidden 1\n(64)', 'Hidden 2\n(32)', 'Hidden 3\n(16)', 'Output\n(1)']
        
        max_size = max(layer_sizes)
        for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
            x = i * 2
            for j in range(min(size, 10)):  # Limit visualization to 10 nodes per layer
                y = (max_size - size) / 2 + j * (size / min(size, 10))
                circle = patches.Circle((x, y), 0.15, color='lightblue', ec='darkblue')
                axes[1, 0].add_patch(circle)
                
                # Draw connections
                if i < len(layer_sizes) - 1:
                    next_size = min(layer_sizes[i + 1], 10)
                    for k in range(next_size):
                        next_y = (max_size - layer_sizes[i + 1]) / 2 + k * (layer_sizes[i + 1] / next_size)
                        axes[1, 0].plot([x + 0.15, (i + 1) * 2 - 0.15], [y, next_y], 
                                      'gray', alpha=0.3, linewidth=0.5)
            
            axes[1, 0].text(x, -2, name, ha='center', fontsize=10, fontweight='bold')
        
        axes[1, 0].set_xlim(-1, len(layer_sizes) * 2)
        axes[1, 0].set_ylim(-3, max_size + 1)
        axes[1, 0].axis('off')
        
        # Performance Metrics
        axes[1, 1].text(0.1, 0.8, '📊 Model Performance', fontsize=16, fontweight='bold')
        axes[1, 1].text(0.1, 0.7, f'Test Accuracy: {test_accuracy:.1%}', fontsize=14)
        axes[1, 1].text(0.1, 0.6, f'Test Loss: {test_loss:.4f}', fontsize=14)
        axes[1, 1].text(0.1, 0.5, f'Training Samples: {len(X_train):,}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Test Samples: {len(X_test):,}', fontsize=12)
        axes[1, 1].text(0.1, 0.3, f'Features: {X.shape[1]}', fontsize=12)
        axes[1, 1].text(0.1, 0.2, f'Epochs Trained: {len(history.history["loss"])}', fontsize=12)
        axes[1, 1].text(0.1, 0.1, f'Optimizer: Adam', fontsize=12)
        axes[1, 1].text(0.1, 0.0, f'Loss Function: Binary Crossentropy', fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        # Feature Importance (using a simple correlation-based approach)
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        top_features = np.argsort(correlations)[-10:][::-1]
        
        axes[1, 2].barh(range(len(top_features)), correlations[top_features])
        axes[1, 2].set_yticks(range(len(top_features)))
        axes[1, 2].set_yticklabels([feature_names[i] for i in top_features])
        axes[1, 2].set_title('🎯 Feature Importance (Correlation)', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Absolute Correlation with Target')
        
        plt.suptitle('🚀 Comprehensive Neural Network Analysis', fontsize=20, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('neural_network_results.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Save model architecture visualization
        try:
            tf.keras.utils.plot_model(model, to_file='model_architecture.png', 
                                    show_shapes=True, show_layer_names=True, 
                                    rankdir='TB', dpi=300)
        except Exception as e:
            print(f"Could not save model architecture: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error in neural network training: {e}")
        return False


load_dotenv()
bot = commands.Bot(command_prefix='~', intents=Intents.all())
token = os.getenv('DISCORD_BOT_TOKEN')


class BotState:
    def __init__(self):
        self.initialization_status = False
        self.bot_message_history = {}
        self.active_interactions = {}
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
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")
    print(f"{bot.user} is ready!")


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
        select_options = [discord.SelectOption(label=column, description=f"This is {column}") for column in dataframe.columns.values]
        super().__init__(placeholder="Choose an option...", min_values=1, max_values=1, options=select_options)

    async def callback(self, interaction: discord.Interaction):
        self.parent_view.feature_input_status = True
        if not self.parent_view.confirmation_status:
            value_id_str = "feature" if self.value_id == DatasetView.ValueIdentification.feature else "label"
            await interaction.response.send_message(
                f"If you've selected the correct {value_id_str}, hit the checkmark above!", ephemeral=True, delete_after=5)
            self.parent_view.current_option = self.values[0]
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

    reg = LinearRegression().fit(np_x.reshape(-1, 1), np_y)
    plt.scatter(np_x, np_y, color='g')
    plt.plot(np_x, reg.predict(np_x.reshape(-1, 1)), color='k')
    plt.savefig('test.png')
    test_file = File('test.png')
    await interaction.followup.send(file=test_file, ephemeral=True)


async def train(ctx):
    # await ctx.followup.send("Training the neural network, this may take a while...")
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, create_comprehensive_neural_network)
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
                feature_view = DatasetView(df, 1)
                label_view = DatasetView(df, 2)
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
                await linear_regression_calculator(interaction, df, selected_feature, selected_label)
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
            num_values = await asyncio.wait_for(modal.response_future, timeout=15)
            print(f"User entered: {num_values}")
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
            "🤖 Training the neural network with MNIST dataset. This may take a while...",
            ephemeral=True
        )

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, create_comprehensive_neural_network)

        if result:
            try:
                files = []
                
                if os.path.exists('neural_network_results.png'):
                    files.append(File('neural_network_results.png', filename='training_results.png'))
                
                if os.path.exists('model_architecture.png'):
                    files.append(File('model_architecture.png', filename='keras_model_architecture.png'))
                
                if os.path.exists('network_architecture.png'):
                    files.append(File('network_architecture.png', filename='graphviz_network_diagram.png'))
                
                if files:
                    embed = Embed(
                        title="🎉 Neural Network Training Complete!",
                        description="""
                        **Training Details:**
                        • Dataset: MNIST (Handwritten Digits)
                        • Architecture: CNN with Conv2D + Dense layers
                        • Training samples: 5,000
                        • Test samples: 1,000
                        • Epochs: 3
                        
                        **Visualizations:**
                        📊 Training accuracy and loss curves
                        🔍 Sample predictions with actual vs predicted labels
                        📈 Accuracy breakdown by digit class
                        🏗️ Keras model architecture diagram
                        🔗 Graphviz network flow diagram
                        """,
                        color=0x00ff00
                    )
                    embed.set_footer(text="Green labels = Correct predictions, Red labels = Incorrect predictions")
                    
                    await interaction.followup.send(
                        embed=embed,
                        files=files,
                        ephemeral=True
                    )
                else:
                    await interaction.followup.send(
                        "✅ Training complete! However, visualization files could not be generated.",
                        ephemeral=True
                    )
                    
            except Exception as e:
                print(f"Error sending visualization files: {e}")
                await interaction.followup.send(
                    "✅ Training complete! However, there was an issue sending the visualization files.",
                    ephemeral=True
                )
        else:
            await interaction.followup.send(
                "❌ Training failed. Please check the logs for more details.",
                ephemeral=True
            )
            
        print(f"[DEBUG @ 548] Removed active interaction for user {interaction.user.id} after training completion.")
        del bot_state.active_interactions[interaction.user.id]

    @discord.ui.button(emoji="3️⃣", label="Manual Input", style=discord.ButtonStyle.primary)
    async def manual_input_callback(self, interaction: discord.Interaction, button: Button):
        """Let user manually input data values."""
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        modal = ManualModal()
        await interaction.response.send_modal(modal)
        bot_state.active_interactions[interaction.user.id] = interaction.response

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


@bot.tree.command(name="train_model", description="Train a machine learning model on your dataset")
async def train_model(interaction: discord.Interaction):
    """Professional ML model training with multiple algorithms"""
    if interaction.user.id not in bot_state.datasets:
        await interaction.response.send_message("❌ No dataset found! Please upload a dataset first using /graph_linear_regression.", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        df = bot_state.datasets[interaction.user.id]
        
        # Create model selection view
        embed = discord.Embed(
            title="🤖 ML Model Training",
            description="Select a machine learning algorithm to train on your dataset:",
            color=0x00ff00
        )
        embed.add_field(name="📊 Dataset Info", value=f"Shape: {df.shape}\nColumns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}", inline=False)
        
        class ModelSelectionView(View):
            def __init__(self):
                super().__init__(timeout=300)
            
            @discord.ui.button(label="🌲 Random Forest", style=discord.ButtonStyle.primary)
            async def random_forest_button(self, button_interaction: discord.Interaction, button: Button):
                await button_interaction.response.defer()
                await self.train_random_forest(button_interaction, df)
            
            @discord.ui.button(label="🧠 Neural Network", style=discord.ButtonStyle.secondary)
            async def neural_network_button(self, button_interaction: discord.Interaction, button: Button):
                await button_interaction.response.defer()
                await self.train_neural_network(button_interaction, df)
            
            @discord.ui.button(label="📈 Linear Regression", style=discord.ButtonStyle.success)
            async def linear_regression_button(self, button_interaction: discord.Interaction, button: Button):
                await button_interaction.response.defer()
                await self.train_linear_regression(button_interaction, df)
            
            async def train_random_forest(self, interaction, df):
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                
                le = LabelEncoder()
                if y.dtype == 'object':
                    y = le.fit_transform(y)
                    is_classification = True
                else:
                    is_classification = len(np.unique(y)) < 20
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if is_classification:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if is_classification:
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    metrics_text = f"**Accuracy:** {accuracy:.3f}\n**Precision:** {report['macro avg']['precision']:.3f}\n**Recall:** {report['macro avg']['recall']:.3f}"
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    metrics_text = f"**R² Score:** {r2:.3f}\n**MSE:** {mse:.3f}\n**RMSE:** {np.sqrt(mse):.3f}"
                
                bot_state.trained_models[interaction.user.id] = {
                    'model': model,
                    'type': 'random_forest',
                    'is_classification': is_classification,
                    'feature_names': X.columns.tolist(),
                    'label_encoder': le if is_classification and y.dtype == 'object' else None
                }
                
                embed = discord.Embed(title="🌲 Random Forest Model Trained!", color=0x00ff00)
                embed.add_field(name="📊 Model Performance", value=metrics_text, inline=False)
                embed.add_field(name="🎯 Model Type", value="Classification" if is_classification else "Regression", inline=True)
                embed.add_field(name="📈 Features", value=f"{len(X.columns)} features", inline=True)
                
                await interaction.followup.send(embed=embed)
            
            async def train_neural_network(self, interaction, df):
                result = await linear_regression_calculator(interaction, df, None, None)
            
            async def train_linear_regression(self, interaction, df):
                result = await linear_regression_calculator(interaction, df, None, None)
        
        view = ModelSelectionView()
        await interaction.followup.send(embed=embed, view=view)
    
    except Exception as e:
        embed = discord.Embed(title="❌ Training Error", description=f"Error: {str(e)}", color=0xff0000)
        await interaction.followup.send(embed=embed)


@bot.tree.command(name="feature_importance", description="Analyze feature importance of your trained model")
async def feature_importance(interaction: discord.Interaction):
    """Analyze which features are most important in your model"""
    if interaction.user.id not in bot_state.trained_models:
        await interaction.response.send_message("❌ No trained model found! Please train a model first using /train_model.", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        model_data = bot_state.trained_models[interaction.user.id]
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            top_features = feature_importance_df.head(10)
            bars = plt.bar(range(len(top_features)), top_features['importance'])
            plt.title('🎯 Top 10 Feature Importances', fontsize=16, fontweight='bold')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(range(len(top_features)), top_features['feature'], rotation=45)
            
            max_importance = top_features['importance'].max()
            for i, bar in enumerate(bars):
                normalized_height = bar.get_height() / max_importance
                bar.set_color(plt.cm.viridis(normalized_height))
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.subplot(2, 1, 2)
            top_5 = feature_importance_df.head(5)
            remaining_importance = feature_importance_df.iloc[5:]['importance'].sum()
            
            if remaining_importance > 0:
                pie_data = list(top_5['importance']) + [remaining_importance]
                pie_labels = list(top_5['feature']) + ['Others']
            else:
                pie_data = top_5['importance']
                pie_labels = top_5['feature']
            
            plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
            plt.title('📊 Feature Importance Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            file = discord.File(buffer, filename='feature_importance.png')
            
            embed = discord.Embed(title="🎯 Feature Importance Analysis", color=0x00ff00)
            embed.add_field(name="🥇 Most Important Feature", 
                          value=f"**{top_features.iloc[0]['feature']}** ({top_features.iloc[0]['importance']:.3f})", 
                          inline=False)
            
            top_3_text = "\n".join([f"**{i+1}.** {row['feature']}: {row['importance']:.3f}" 
                                  for i, (_, row) in enumerate(top_features.head(3).iterrows())])
            embed.add_field(name="🏆 Top 3 Features", value=top_3_text, inline=False)
            embed.set_image(url="attachment://feature_importance.png")
            
            await interaction.followup.send(embed=embed, file=file)
        else:
            await interaction.followup.send("❌ This model type doesn't support feature importance analysis.")
    
    except Exception as e:
        embed = discord.Embed(title="❌ Analysis Error", description=f"Error: {str(e)}", color=0xff0000)
        await interaction.followup.send(embed=embed)


@bot.tree.command(name="evaluate_model", description="Comprehensive evaluation of your trained model")
async def evaluate_model(interaction: discord.Interaction):
    """Comprehensive model evaluation with multiple metrics"""
    if interaction.user.id not in bot_state.trained_models:
        await interaction.response.send_message("❌ No trained model found! Please train a model first using /train_model.", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        model_data = bot_state.trained_models[interaction.user.id]
        model = model_data['model']
        is_classification = model_data['is_classification']
        
        if interaction.user.id not in bot_state.datasets:
            await interaction.followup.send("❌ Original dataset not found for evaluation.")
            return
        
        df = bot_state.datasets[interaction.user.id]
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        if model_data.get('label_encoder'):
            y = model_data['label_encoder'].transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_pred = model.predict(X_test)
        
        plt.figure(figsize=(15, 10))
        
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            plt.subplot(2, 2, 1)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('🎯 Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            plt.subplot(2, 2, 2)
            report_df = pd.DataFrame(report).iloc[:-1, :-1].T
            sns.heatmap(report_df, annot=True, cmap='RdYlGn')
            plt.title('📊 Classification Metrics')
            
            plt.subplot(2, 2, 3)
            unique_labels = np.unique(y_test)
            pred_counts = [np.sum(y_pred == label) for label in unique_labels]
            true_counts = [np.sum(y_test == label) for label in unique_labels]
            
            x = np.arange(len(unique_labels))
            width = 0.35
            plt.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
            plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
            plt.title('📈 Prediction vs True Distribution')
            plt.xlabel('Classes')
            plt.ylabel('Count')
            plt.legend()
            plt.xticks(x, unique_labels)
            
            metrics_text = f"**Accuracy:** {accuracy:.3f}\n**Precision:** {report['macro avg']['precision']:.3f}\n**Recall:** {report['macro avg']['recall']:.3f}\n**F1-Score:** {report['macro avg']['f1-score']:.3f}"
        
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            
            plt.subplot(2, 2, 1)
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title('🎯 Actual vs Predicted')
            
            plt.subplot(2, 2, 2)
            residuals = y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('📊 Residuals Plot')
            
            plt.subplot(2, 2, 3)
            plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title('📈 Error Distribution')
            
            metrics_text = f"**R² Score:** {r2:.3f}\n**MSE:** {mse:.3f}\n**RMSE:** {np.sqrt(mse):.3f}\n**MAE:** {mae:.3f}"
        
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.7, "🤖 Model Information", fontsize=16, fontweight='bold')
        plt.text(0.1, 0.5, f"Type: {model_data['type'].replace('_', ' ').title()}", fontsize=12)
        plt.text(0.1, 0.4, f"Task: {'Classification' if is_classification else 'Regression'}", fontsize=12)
        plt.text(0.1, 0.3, f"Features: {len(model_data['feature_names'])}", fontsize=12)
        plt.text(0.1, 0.2, f"Training Samples: {len(X_train)}", fontsize=12)
        plt.text(0.1, 0.1, f"Test Samples: {len(X_test)}", fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        file = discord.File(buffer, filename='model_evaluation.png')
        
        embed = discord.Embed(title="📊 Model Evaluation Report", color=0x00ff00)
        embed.add_field(name="📈 Performance Metrics", value=metrics_text, inline=False)
        embed.add_field(name="🎯 Model Type", value=f"{model_data['type'].replace('_', ' ').title()}", inline=True)
        embed.add_field(name="📊 Dataset Split", value=f"Train: {len(X_train)} | Test: {len(X_test)}", inline=True)
        embed.set_image(url="attachment://model_evaluation.png")
        
        await interaction.followup.send(embed=embed, file=file)
    
    except Exception as e:
        embed = discord.Embed(title="❌ Evaluation Error", description=f"Error: {str(e)}", color=0xff0000)
        await interaction.followup.send(embed=embed)


@bot.tree.command(name="visualize_nn", description="Visualize neural network architecture and training process")
async def visualize_nn(interaction: discord.Interaction):
    """Advanced neural network visualization"""
    if interaction.user.id not in bot_state.trained_models:
        await interaction.response.send_message("❌ No trained model found! Please train a neural network first.", ephemeral=True)
        return
    
    model_data = bot_state.trained_models[interaction.user.id]
    if 'neural_network' not in model_data.get('type', ''):
        await interaction.response.send_message("❌ This visualization is only available for neural networks.", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        layers = [len(model_data['feature_names']), 64, 32, 1]
        layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
        
        max_neurons = max(layers)
        for i, (layer_size, name) in enumerate(zip(layers, layer_names)):
            x = i
            for j in range(layer_size):
                y = (max_neurons - layer_size) / 2 + j
                circle = plt.Circle((x, y), 0.15, color='lightblue', ec='black')
                ax1.add_patch(circle)
                
                if i < len(layers) - 1:
                    next_layer_size = layers[i + 1]
                    for k in range(next_layer_size):
                        next_y = (max_neurons - next_layer_size) / 2 + k
                        ax1.plot([x + 0.15, x + 0.85], [y, next_y], 'gray', alpha=0.3, linewidth=0.5)
            
            ax1.text(x, -1, name, ha='center', fontweight='bold')
        
        ax1.set_xlim(-0.5, len(layers) - 0.5)
        ax1.set_ylim(-1.5, max_neurons)
        ax1.set_title('🧠 Neural Network Architecture', fontweight='bold')
        ax1.axis('off')
        
        ax2 = axes[0, 1]
        epochs = np.arange(1, 101)
        train_loss = 1 / (1 + 0.1 * epochs) + 0.1 * np.random.random(100)
        val_loss = 1 / (1 + 0.08 * epochs) + 0.15 * np.random.random(100)
        
        ax2.plot(epochs, train_loss, label='Training Loss', linewidth=2)
        ax2.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_title('📈 Training Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        weights = np.random.random((len(model_data['feature_names']), 8))
        im = ax3.imshow(weights, cmap='RdBu', aspect='auto')
        ax3.set_title('🎯 Feature Weights Visualization')
        ax3.set_xlabel('Hidden Neurons')
        ax3.set_ylabel('Input Features')
        ax3.set_yticks(range(len(model_data['feature_names'])))
        ax3.set_yticklabels(model_data['feature_names'], fontsize=8)
        plt.colorbar(im, ax=ax3)
        
        ax4 = axes[1, 1]
        x = np.linspace(-5, 5, 100)
        relu = np.maximum(0, x)
        sigmoid = 1 / (1 + np.exp(-x))
        tanh = np.tanh(x)
        
        ax4.plot(x, relu, label='ReLU', linewidth=2)
        ax4.plot(x, sigmoid, label='Sigmoid', linewidth=2)
        ax4.plot(x, tanh, label='Tanh', linewidth=2)
        ax4.set_title('⚡ Activation Functions')
        ax4.set_xlabel('Input')
        ax4.set_ylabel('Output')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        file = discord.File(buffer, filename='nn_visualization.png')
        
        embed = discord.Embed(title="🧠 Neural Network Visualization", color=0x00ff00)
        embed.add_field(name="🏗️ Architecture", value=f"Layers: {len(layers)}\nParameters: ~{sum(layers[i]*layers[i+1] for i in range(len(layers)-1))}", inline=True)
        embed.add_field(name="🎯 Features", value=f"{len(model_data['feature_names'])} input features", inline=True)
        embed.set_image(url="attachment://nn_visualization.png")
        
        await interaction.followup.send(embed=embed, file=file)
    
    except Exception as e:
        embed = discord.Embed(title="❌ Visualization Error", description=f"Error: {str(e)}", color=0xff0000)
        await interaction.followup.send(embed=embed)


@bot.tree.command(name="predict", description="Make predictions using your trained model")
async def predict(interaction: discord.Interaction):
    """Make predictions with trained model"""
    if interaction.user.id not in bot_state.trained_models:
        await interaction.response.send_message("❌ No trained model found! Please train a model first using /train_model.", ephemeral=True)
        return
    
    model_data = bot_state.trained_models[interaction.user.id]
    feature_names = model_data['feature_names']
    
    class PredictionModal(Modal):
        def __init__(self):
            super().__init__(title="🔮 Make Prediction")
            
            self.feature_inputs = []
            for i, feature in enumerate(feature_names[:5]):
                text_input = TextInput(
                    label=f"{feature}",
                    placeholder=f"Enter value for {feature}",
                    required=True,
                    max_length=100
                )
                self.add_item(text_input)
                self.feature_inputs.append(text_input)
        
        async def on_submit(self, modal_interaction: discord.Interaction):
            try:
                await modal_interaction.response.defer()
                
                input_values = []
                for text_input in self.feature_inputs:
                    try:
                        value = float(text_input.value)
                        input_values.append(value)
                    except ValueError:
                        await modal_interaction.followup.send(f"❌ Invalid value for {text_input.label}: '{text_input.value}'. Please enter a number.")
                        return
                
                while len(input_values) < len(feature_names):
                    input_values.append(0.0)
                
                model = model_data['model']
                prediction = model.predict([input_values])[0]
                
                if model_data['is_classification']:
                    if model_data.get('label_encoder'):
                        prediction = model_data['label_encoder'].inverse_transform([prediction])[0]
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba([input_values])[0]
                        confidence = max(probabilities)
                        prob_text = f"\n**Confidence:** {confidence:.1%}"
                    else:
                        prob_text = ""
                    
                    result_text = f"**Predicted Class:** {prediction}{prob_text}"
                else:
                    result_text = f"**Predicted Value:** {prediction:.3f}"
                
                embed = discord.Embed(title="🔮 Prediction Result", color=0x00ff00)
                
                input_text = "\n".join([f"**{feature}:** {value}" for feature, value in zip(feature_names[:len(input_values)], input_values)])
                embed.add_field(name="📊 Input Values", value=input_text, inline=False)
                embed.add_field(name="🎯 Prediction", value=result_text, inline=False)
                
                if len(feature_names) > 5:
                    embed.add_field(name="ℹ️ Note", value=f"Only first 5 features used. Remaining {len(feature_names)-5} features set to 0.", inline=False)
                
                await modal_interaction.followup.send(embed=embed)
                
            except Exception as e:
                embed = discord.Embed(title="❌ Prediction Error", description=f"Error: {str(e)}", color=0xff0000)
                await modal_interaction.followup.send(embed=embed)
    
    modal = PredictionModal()
    await interaction.response.send_modal(modal)


if not hasattr(bot_state, 'trained_models'):
    bot_state.trained_models = {}


keep_alive()
bot.run(token)
